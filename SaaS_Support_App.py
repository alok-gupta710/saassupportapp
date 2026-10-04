"""
SaaS Support Multi-Agent Webhook (CrewAI + FastAPI) - Render deployment.

Start command on Render:
    uvicorn main:app --host 0.0.0.0 --port $PORT

Environment variables:
    OPENAI_API_KEY          optional; if missing, runs in deterministic simulation mode
    SUPPORT_CREW_MODEL      optional; default gpt-4o-mini
    WEBHOOK_SHARED_SECRET   optional but strongly recommended; callers must send X-Webhook-Secret
    PYTHON_VERSION          set to 3.10.13 on Render (CrewAI 0.80.0 target)
"""

import os

# Must be set before crewai is imported anywhere.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import json
import uuid
import time
import logging
import concurrent.futures
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("webhook_app")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL_NAME = os.getenv("SUPPORT_CREW_MODEL", "gpt-4o-mini")
WEBHOOK_SHARED_SECRET = os.getenv("WEBHOOK_SHARED_SECRET", "").strip()
SIMULATION_MODE = len(OPENAI_API_KEY) == 0

# ---------------------------------------------------------------------------
# CrewAI import (+ litellm/pydantic compatibility patch)
# ---------------------------------------------------------------------------
try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.flow.flow import Flow, start, listen, router
    CREWAI_IMPORT_OK = True
except Exception as import_error:
    logger.warning("crewai import failed (%s). Forcing simulation mode.", import_error)
    CREWAI_IMPORT_OK = False

if CREWAI_IMPORT_OK:
    try:
        from litellm.types.llms.openai import ChatCompletionReasoningSummaryTextBlock
        from litellm.types.utils import Message as _LiteLLMMessage

        _LiteLLMMessage.model_rebuild(
            force=True,
            _types_namespace={"ChatCompletionReasoningSummaryTextBlock": ChatCompletionReasoningSummaryTextBlock},
        )
        logger.info("litellm/pydantic compatibility patch applied.")
    except Exception as compat_error:
        logger.info("litellm/pydantic compatibility patch skipped: %s", compat_error)

USE_LIVE_LLM = CREWAI_IMPORT_OK and not SIMULATION_MODE

if not CREWAI_IMPORT_OK:
    # Minimal stand-ins so the module still imports and simulation mode works.
    class Flow:  # type: ignore
        def __class_getitem__(cls, item):
            return cls

    def start(*a, **k):  # type: ignore
        return lambda f: f

    def listen(*a, **k):  # type: ignore
        return lambda f: f

    def router(*a, **k):  # type: ignore
        return lambda f: f


# ---------------------------------------------------------------------------
# 1. Handoff contracts
# ---------------------------------------------------------------------------
class IssueCategory(str, Enum):
    LOGIN_AUTH = "login_auth"
    API_ERROR = "api_error"
    FEATURE_USAGE = "feature_usage"
    INTEGRATION = "integration"
    BILLING = "billing"
    OUTAGE_PERFORMANCE = "outage_performance"
    SECURITY_DATA = "security_data"
    OTHER = "other"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    PARTIAL = "partial"
    UNRESOLVED = "unresolved"


class DiagnosisResult(BaseModel):
    issue_category: IssueCategory
    symptoms_summary: str = Field(..., description="1-2 sentence neutral summary of what the customer reported")
    error_codes: List[str] = Field(default_factory=list)
    severity: SeverityLevel
    urgency: UrgencyLevel
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("confidence")
    @classmethod
    def _round_confidence(cls, v: float) -> float:
        return round(v, 2)


class KnowledgeResult(BaseModel):
    known_issue: bool
    kb_article_ids: List[str] = Field(default_factory=list)
    root_cause_hypothesis: str
    applicable_policy: str
    suggested_fix_type: str = Field(..., description="self_serve, config_change, known_bug_workaround, no_known_fix")
    reasoning_confidence: float = Field(..., ge=0.0, le=1.0)


class TroubleshootingResult(BaseModel):
    customer_response: str
    steps_provided: List[str]
    resolution_status: ResolutionStatus
    customer_action_required: bool


class EscalationDecision(BaseModel):
    escalate: bool
    risk_score: float = Field(..., ge=0.0, le=1.0)
    triggered_rules: List[str]
    rationale: str


class EngineeringTicket(BaseModel):
    ticket_id: str
    priority: str
    title: str
    summary_for_engineering: str
    reproduction_context: str
    affected_component: str
    customer_impact: str
    recommended_next_step: str


class SupportFlowState(BaseModel):
    flow_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    customer_tier: str = "standard"
    customer_reported_urgency: str = "normal"

    diagnosis: Optional[DiagnosisResult] = None
    knowledge: Optional[KnowledgeResult] = None
    troubleshooting: Optional[TroubleshootingResult] = None
    escalation_decision: Optional[EscalationDecision] = None
    engineering_ticket: Optional[EngineeringTicket] = None

    final_customer_message: str = ""
    stage_errors: List[str] = Field(default_factory=list)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# 2. Non-RAG knowledge base
# ---------------------------------------------------------------------------
PRODUCT_KNOWLEDGE_BASE = {
    "login_auth": {
        "kb_article_ids": ["KB-101", "KB-104"],
        "common_root_causes": [
            "SSO token expired after password rotation",
            "MFA device out of sync / clock drift",
            "Account locked after repeated failed attempts (security policy)",
        ],
        "policy": "SSO_TOKEN_EXPIRY_POLICY",
        "default_fix_type": "self_serve",
    },
    "api_error": {
        "kb_article_ids": ["KB-210", "KB-215"],
        "common_root_causes": [
            "API key expired or rotated without updating client config",
            "Rate limit exceeded (429) due to burst traffic",
            "Malformed request payload / schema mismatch after a version bump",
        ],
        "policy": "API_RATE_LIMIT_AND_KEY_ROTATION_POLICY",
        "default_fix_type": "config_change",
    },
    "feature_usage": {
        "kb_article_ids": ["KB-300"],
        "common_root_causes": [
            "Feature gated behind a plan tier the customer is not on",
            "Feature flag not yet enabled for the account",
            "User has insufficient role/permission for the action",
        ],
        "policy": "FEATURE_ENTITLEMENT_POLICY",
        "default_fix_type": "self_serve",
    },
    "integration": {
        "kb_article_ids": ["KB-410", "KB-418"],
        "common_root_causes": [
            "Webhook endpoint unreachable / customer-side firewall change",
            "OAuth scope changed by the third-party app",
            "Version mismatch between our SDK and the partner platform",
        ],
        "policy": "THIRD_PARTY_INTEGRATION_SUPPORT_POLICY",
        "default_fix_type": "config_change",
    },
    "billing": {
        "kb_article_ids": ["KB-500"],
        "common_root_causes": [
            "Card expired / payment declined",
            "Plan downgrade removed access to a previously used feature",
        ],
        "policy": "BILLING_SELF_SERVICE_POLICY",
        "default_fix_type": "self_serve",
    },
    "outage_performance": {
        "kb_article_ids": [],
        "common_root_causes": ["No matching known issue - possible active incident"],
        "policy": "ACTIVE_INCIDENT_ESCALATION_POLICY",
        "default_fix_type": "no_known_fix",
    },
    "security_data": {
        "kb_article_ids": [],
        "common_root_causes": ["No matching known issue - potential security/data-integrity event"],
        "policy": "SECURITY_INCIDENT_ESCALATION_POLICY",
        "default_fix_type": "no_known_fix",
    },
    "other": {
        "kb_article_ids": [],
        "common_root_causes": ["Uncategorized - needs human judgement"],
        "policy": "GENERAL_SUPPORT_POLICY",
        "default_fix_type": "no_known_fix",
    },
}


# ---------------------------------------------------------------------------
# 3. Escalation policy engine (deterministic)
# ---------------------------------------------------------------------------
RULE_WEIGHTS = {
    "R1_low_diagnosis_confidence": 0.30,
    "R2_high_or_critical_severity": 0.35,
    "R3_high_risk_category": 0.45,
    "R4_low_knowledge_confidence": 0.25,
    "R5_no_known_fix": 0.30,
    "R6_troubleshooting_unresolved": 0.40,
    "R7_high_urgency_not_resolved": 0.25,
    "R8_enterprise_partial_resolution": 0.40,
}

DIAGNOSIS_CONFIDENCE_THRESHOLD = 0.55
KNOWLEDGE_CONFIDENCE_THRESHOLD = 0.50
HIGH_RISK_CATEGORIES = {"security_data", "outage_performance"}
ESCALATION_SCORE_THRESHOLD = 0.35


def _val(x):
    return x.value if hasattr(x, "value") else x


def evaluate_escalation_policy(state: SupportFlowState) -> EscalationDecision:
    if state.diagnosis is None or state.knowledge is None or state.troubleshooting is None:
        raise ValueError("Cannot evaluate escalation policy: an upstream stage did not complete.")

    triggered = []
    if state.diagnosis.confidence < DIAGNOSIS_CONFIDENCE_THRESHOLD:
        triggered.append("R1_low_diagnosis_confidence")
    if _val(state.diagnosis.severity) in ("high", "critical"):
        triggered.append("R2_high_or_critical_severity")
    if _val(state.diagnosis.issue_category) in HIGH_RISK_CATEGORIES:
        triggered.append("R3_high_risk_category")
    if state.knowledge.reasoning_confidence < KNOWLEDGE_CONFIDENCE_THRESHOLD:
        triggered.append("R4_low_knowledge_confidence")
    if state.knowledge.suggested_fix_type == "no_known_fix":
        triggered.append("R5_no_known_fix")

    resolution_status = _val(state.troubleshooting.resolution_status)
    if resolution_status == "unresolved":
        triggered.append("R6_troubleshooting_unresolved")
    if _val(state.diagnosis.urgency) == "high" and resolution_status != "resolved":
        triggered.append("R7_high_urgency_not_resolved")
    if state.customer_tier == "enterprise" and resolution_status == "partial":
        triggered.append("R8_enterprise_partial_resolution")

    risk_score = min(1.0, sum(RULE_WEIGHTS[r] for r in triggered))
    escalate = risk_score >= ESCALATION_SCORE_THRESHOLD

    if triggered:
        rationale = ("Escalating because: " if escalate else "Below escalation threshold despite: ") + "; ".join(triggered)
    else:
        rationale = "No risk/uncertainty/urgency rules triggered - safe to resolve via customer-facing response."

    return EscalationDecision(
        escalate=escalate,
        risk_score=round(risk_score, 2),
        triggered_rules=triggered,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# 4. CrewAI agents
# ---------------------------------------------------------------------------
def get_llm():
    try:
        return LLM(model=LLM_MODEL_NAME, temperature=0.2)
    except Exception as llm_error:
        raise RuntimeError(f"Could not initialize LLM '{LLM_MODEL_NAME}': {llm_error}") from llm_error


def build_diagnosis_agent(llm):
    return Agent(
        role="Technical Issue Diagnosis Specialist",
        goal=("Read a raw customer support message and produce a precise, structured diagnosis: "
              "issue category, severity, urgency, and a calibrated confidence score. "
              "Never invent facts the customer did not state."),
        backstory=("You are a senior SaaS support triage engineer with years of experience "
                   "reading noisy customer messages and quickly classifying the underlying "
                   "technical issue without overstating certainty."),
        llm=llm, verbose=False, allow_delegation=False,
    )


def build_knowledge_agent(llm):
    return Agent(
        role="Product Knowledge Reasoning Analyst",
        goal=("Given a structured diagnosis and a curated internal knowledge-base entry for that "
              "issue category, reason (without any external search or retrieval) about the most "
              "likely root cause and which internal policy applies."),
        backstory=("You are an internal product knowledge analyst. You only reason over the "
                   "knowledge base facts you are given in the task context - you never browse "
                   "the web and never invent a KB article ID that wasn't provided to you."),
        llm=llm, verbose=False, allow_delegation=False,
    )


def build_troubleshooting_agent(llm):
    return Agent(
        role="Troubleshooting Response Writer",
        goal=("Write a clear, empathetic, step-by-step customer response based on the diagnosis "
              "and knowledge-base hypothesis, and honestly report whether this is likely to fully "
              "resolve the issue, partially resolve it, or leave it unresolved."),
        backstory=("You are a customer-facing support agent who is rewarded for genuinely solving "
                   "problems, not for appearing helpful. You are honest when a fix is not guaranteed."),
        llm=llm, verbose=False, allow_delegation=False,
    )


def build_escalation_agent(llm):
    return Agent(
        role="Engineering Escalation Coordinator",
        goal=("Convert a flagged, unresolved support case into a precise, engineering-ready ticket: "
              "clear title, reproduction context, affected component, customer impact, and a priority "
              "level, so engineers can act without going back to the customer."),
        backstory=("You are the bridge between support and engineering. Engineers trust your tickets "
                   "because they are specific and never padded with speculation."),
        llm=llm, verbose=False, allow_delegation=False,
    )


# ---------------------------------------------------------------------------
# 5. CrewAI tasks
# ---------------------------------------------------------------------------
def build_diagnosis_task(agent, query: str):
    return Task(
        description=(
            f"Customer message:\n\"\"\"{query}\"\"\"\n\n"
            "Classify this into exactly one issue_category from: "
            "login_auth, api_error, feature_usage, integration, billing, "
            "outage_performance, security_data, other.\n"
            "Extract any explicit error codes mentioned (empty list if none).\n"
            "Assign severity (low/medium/high/critical) based on business impact implied by the "
            "message, and urgency (low/medium/high) based on time-sensitivity language.\n"
            "Give a confidence score between 0 and 1 for your own classification - be honest, "
            "lower it if the message is vague or could fit multiple categories."
        ),
        expected_output="A DiagnosisResult JSON object matching the provided schema.",
        agent=agent,
        output_pydantic=DiagnosisResult,
    )


def build_knowledge_task(agent, diagnosis: DiagnosisResult):
    kb_entry = PRODUCT_KNOWLEDGE_BASE.get(_val(diagnosis.issue_category), PRODUCT_KNOWLEDGE_BASE["other"])
    return Task(
        description=(
            f"Diagnosis handed off from the Diagnosis Agent:\n{diagnosis.model_dump_json(indent=2)}\n\n"
            f"Internal (non-RAG) knowledge base entry for this category:\n{json.dumps(kb_entry, indent=2)}\n\n"
            "Pick the single most likely root_cause_hypothesis from common_root_causes (or write a "
            "closely related one only if none fit AND explain why in the hypothesis text). "
            "Set known_issue=true only if a common_root_causes entry plausibly matches. "
            "Copy kb_article_ids from the entry as-is (do not invent new IDs). "
            "Set applicable_policy to the policy field given. "
            "Choose suggested_fix_type from: self_serve, config_change, known_bug_workaround, no_known_fix "
            "(use default_fix_type unless the symptoms clearly point elsewhere). "
            "Give reasoning_confidence between 0 and 1 for how well the KB entry matches these symptoms."
        ),
        expected_output="A KnowledgeResult JSON object matching the provided schema.",
        agent=agent,
        output_pydantic=KnowledgeResult,
    )


def build_troubleshooting_task(agent, diagnosis: DiagnosisResult, knowledge: KnowledgeResult):
    return Task(
        description=(
            f"Diagnosis:\n{diagnosis.model_dump_json(indent=2)}\n\n"
            f"Knowledge reasoning:\n{knowledge.model_dump_json(indent=2)}\n\n"
            "Write a customer_response (plain text, 3-6 sentences, empathetic but efficient) that "
            "acts on the root_cause_hypothesis. List concrete steps_provided (1-5 short imperative "
            "steps). Set customer_action_required=true if the customer must do something themselves.\n"
            "Set resolution_status honestly:\n"
            "- 'resolved' only if suggested_fix_type is self_serve or config_change AND you are "
            "confident these steps fix the root cause,\n"
            "- 'partial' if the steps help but do not fully guarantee a fix,\n"
            "- 'unresolved' if suggested_fix_type is no_known_fix or known_bug_workaround with no "
            "reliable customer-side workaround."
        ),
        expected_output="A TroubleshootingResult JSON object matching the provided schema.",
        agent=agent,
        output_pydantic=TroubleshootingResult,
    )


def build_escalation_task(agent, state: SupportFlowState):
    return Task(
        description=(
            f"This case was flagged for escalation by the policy engine:\n"
            f"{state.escalation_decision.model_dump_json(indent=2)}\n\n"
            f"Diagnosis:\n{state.diagnosis.model_dump_json(indent=2)}\n\n"
            f"Knowledge reasoning:\n{state.knowledge.model_dump_json(indent=2)}\n\n"
            f"Troubleshooting attempt:\n{state.troubleshooting.model_dump_json(indent=2)}\n\n"
            f"Customer tier: {state.customer_tier}\n\n"
            "Write an EngineeringTicket: a short specific title, a summary_for_engineering with "
            "only facts already given (no speculation), reproduction_context describing what the "
            "customer experienced, affected_component (best guess system name, e.g. 'auth-service', "
            "'public-api-gateway', 'webhooks'), customer_impact in one sentence, "
            "recommended_next_step for the engineering team, and priority as P1 (critical/outage), "
            "P2 (high risk_score >= 0.7), P3 (moderate), or P4 (low, informational)."
        ),
        expected_output="An EngineeringTicket JSON object matching the provided schema, with ticket_id left as 'TBD' (the system assigns the real ID).",
        agent=agent,
        output_pydantic=EngineeringTicket,
    )


# ---------------------------------------------------------------------------
# 6. Deterministic simulators (no-key mode + per-stage fallback)
# ---------------------------------------------------------------------------
def simulate_diagnosis(query: str, customer_reported_urgency: str) -> DiagnosisResult:
    q = query.lower()

    if any(w in q for w in ["breach", "leaked", "unauthorized access", "exposed our data", "data loss",
                             "another company's", "someone else's data", "other customer's data",
                             "seeing another", "wrong account's data"]):
        category, severity, confidence = IssueCategory.SECURITY_DATA, SeverityLevel.CRITICAL, 0.6
    elif any(w in q for w in ["down", "outage", "all users", "everyone is affected", "cannot access the platform"]):
        category, severity, confidence = IssueCategory.OUTAGE_PERFORMANCE, SeverityLevel.CRITICAL, 0.65
    elif any(w in q for w in ["login", "log in", "sign in", "password", "mfa", "2fa", "sso"]):
        category, severity, confidence = IssueCategory.LOGIN_AUTH, SeverityLevel.MEDIUM, 0.82
    elif any(w in q for w in ["api", "401", "403", "429", "endpoint", "webhook", "status code"]):
        category, severity, confidence = IssueCategory.API_ERROR, SeverityLevel.MEDIUM, 0.78
    elif any(w in q for w in ["integrat", "zapier", "salesforce", "slack app", "third-party", "oauth"]):
        category, severity, confidence = IssueCategory.INTEGRATION, SeverityLevel.MEDIUM, 0.7
    elif any(w in q for w in ["invoice", "charged", "billing", "payment", "card declined", "refund"]):
        category, severity, confidence = IssueCategory.BILLING, SeverityLevel.LOW, 0.8
    elif any(w in q for w in ["how do i", "how to", "where is the button", "can't find", "not sure how"]):
        category, severity, confidence = IssueCategory.FEATURE_USAGE, SeverityLevel.LOW, 0.75
    else:
        category, severity, confidence = IssueCategory.OTHER, SeverityLevel.LOW, 0.35

    urgency = UrgencyLevel.HIGH if (customer_reported_urgency == "urgent" or severity in
                                     (SeverityLevel.HIGH, SeverityLevel.CRITICAL)) else UrgencyLevel.MEDIUM
    if severity == SeverityLevel.LOW and customer_reported_urgency != "urgent":
        urgency = UrgencyLevel.LOW

    error_codes = [tok.strip(",.") for tok in query.split()
                   if tok.strip(",.").isdigit() and len(tok.strip(",.")) == 3]

    return DiagnosisResult(
        issue_category=category,
        symptoms_summary=(query[:180] + ("..." if len(query) > 180 else "")),
        error_codes=error_codes,
        severity=severity,
        urgency=urgency,
        confidence=confidence,
    )


def simulate_knowledge(diagnosis: DiagnosisResult) -> KnowledgeResult:
    entry = PRODUCT_KNOWLEDGE_BASE.get(_val(diagnosis.issue_category), PRODUCT_KNOWLEDGE_BASE["other"])
    known = len(entry["kb_article_ids"]) > 0 and diagnosis.confidence >= 0.5
    return KnowledgeResult(
        known_issue=known,
        kb_article_ids=entry["kb_article_ids"] if known else [],
        root_cause_hypothesis=entry["common_root_causes"][0],
        applicable_policy=entry["policy"],
        suggested_fix_type=entry["default_fix_type"],
        reasoning_confidence=0.75 if known else 0.35,
    )


def simulate_troubleshooting(diagnosis: DiagnosisResult, knowledge: KnowledgeResult) -> TroubleshootingResult:
    fix_type = knowledge.suggested_fix_type
    if fix_type == "self_serve":
        steps = ["Clear your browser cache and cookies", "Try logging in again in an incognito window",
                 "If using MFA, resync your authenticator app's time"]
        status_, response = ResolutionStatus.RESOLVED, (
            f"Thanks for the details! This looks like {knowledge.root_cause_hypothesis.lower()}. "
            "Please try the steps below - this resolves the issue for the large majority of similar cases."
        )
    elif fix_type == "config_change":
        steps = ["Verify your API key/config against the latest integration docs",
                 "Regenerate the credential if it was rotated recently",
                 "Retry the request and check the response headers for rate-limit details"]
        status_, response = ResolutionStatus.PARTIAL, (
            f"This looks related to {knowledge.root_cause_hypothesis.lower()}. "
            "The steps below should get you unblocked, though please confirm it resolves things on your end."
        )
    else:
        steps = ["We've logged the details you shared", "Our team will investigate the underlying cause"]
        status_, response = ResolutionStatus.UNRESOLVED, (
            "Thanks for reporting this - based on what you've shared, this does not match a known "
            "self-service fix, so I'm escalating it to our engineering team rather than have you try "
            "workarounds that may not resolve it."
        )

    return TroubleshootingResult(
        customer_response=response,
        steps_provided=steps,
        resolution_status=status_,
        customer_action_required=(fix_type in ("self_serve", "config_change")),
    )


def simulate_escalation_ticket(state: SupportFlowState) -> EngineeringTicket:
    cat_value = _val(state.diagnosis.issue_category)
    component_map = {
        "login_auth": "auth-service", "api_error": "public-api-gateway",
        "integration": "webhooks", "outage_performance": "core-platform",
        "security_data": "security-and-data-integrity", "billing": "billing-service",
        "feature_usage": "app-frontend", "other": "unassigned",
    }
    risk = state.escalation_decision.risk_score
    priority = "P1" if _val(state.diagnosis.severity) == "critical" else \
               "P2" if risk >= 0.7 else "P3" if risk >= 0.4 else "P4"

    return EngineeringTicket(
        ticket_id="TBD",
        priority=priority,
        title=f"[{cat_value}] {state.diagnosis.symptoms_summary[:70]}",
        summary_for_engineering=(
            f"Customer report: {state.diagnosis.symptoms_summary} "
            f"Suspected cause: {state.knowledge.root_cause_hypothesis} "
            f"(policy: {state.knowledge.applicable_policy})."
        ),
        reproduction_context=state.query,
        affected_component=component_map.get(cat_value, "unassigned"),
        customer_impact=f"{state.customer_tier} customer, severity={_val(state.diagnosis.severity)}, "
                         f"urgency={_val(state.diagnosis.urgency)}, resolution_status="
                         f"{_val(state.troubleshooting.resolution_status)}.",
        recommended_next_step="Investigate root cause; support has already ruled out known self-serve fixes.",
    )


# ---------------------------------------------------------------------------
# 7. Orchestrator: CrewAI Flow
# ---------------------------------------------------------------------------
def _summarize_error(exc: Exception, max_len: int = 160) -> str:
    text = str(exc).strip()
    first_line = text.splitlines()[0] if text else ""
    if not first_line:
        first_line = exc.__class__.__name__
    if len(first_line) > max_len:
        first_line = first_line[:max_len].rstrip() + "..."
    return f"{exc.__class__.__name__}: {first_line}"


def _run_single_agent_task(agent, task):
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    return crew.kickoff().pydantic


class SupportWorkflowFlow(Flow[SupportFlowState]):

    @start()
    def diagnose_issue(self):
        try:
            if not USE_LIVE_LLM:
                raise RuntimeError("SIMULATION_MODE active - skipping live LLM call by design.")
            agent = build_diagnosis_agent(get_llm())
            result = _run_single_agent_task(agent, build_diagnosis_task(agent, self.state.query))
            if result is None:
                raise ValueError("Diagnosis task did not return valid structured output.")
            self.state.diagnosis = result
        except Exception as stage_error:
            self.state.stage_errors.append(f"diagnose_issue: {_summarize_error(stage_error)}")
            self.state.diagnosis = simulate_diagnosis(self.state.query, self.state.customer_reported_urgency)
        return "diagnosis_complete"

    @listen(diagnose_issue)
    def reason_with_knowledge(self):
        try:
            if not USE_LIVE_LLM:
                raise RuntimeError("SIMULATION_MODE active - skipping live LLM call by design.")
            agent = build_knowledge_agent(get_llm())
            result = _run_single_agent_task(agent, build_knowledge_task(agent, self.state.diagnosis))
            if result is None:
                raise ValueError("Knowledge task did not return valid structured output.")
            self.state.knowledge = result
        except Exception as stage_error:
            self.state.stage_errors.append(f"reason_with_knowledge: {_summarize_error(stage_error)}")
            self.state.knowledge = simulate_knowledge(self.state.diagnosis)
        return "knowledge_complete"

    @listen(reason_with_knowledge)
    def attempt_troubleshooting(self):
        try:
            if not USE_LIVE_LLM:
                raise RuntimeError("SIMULATION_MODE active - skipping live LLM call by design.")
            agent = build_troubleshooting_agent(get_llm())
            result = _run_single_agent_task(
                agent, build_troubleshooting_task(agent, self.state.diagnosis, self.state.knowledge))
            if result is None:
                raise ValueError("Troubleshooting task did not return valid structured output.")
            self.state.troubleshooting = result
        except Exception as stage_error:
            self.state.stage_errors.append(f"attempt_troubleshooting: {_summarize_error(stage_error)}")
            self.state.troubleshooting = simulate_troubleshooting(self.state.diagnosis, self.state.knowledge)
        return "troubleshooting_complete"

    @router(attempt_troubleshooting)
    def decide_escalation(self):
        try:
            decision = evaluate_escalation_policy(self.state)
        except Exception as policy_error:
            self.state.stage_errors.append(f"decide_escalation: {_summarize_error(policy_error)}")
            decision = EscalationDecision(
                escalate=True, risk_score=1.0, triggered_rules=["POLICY_ENGINE_ERROR"],
                rationale=f"Escalation policy engine raised an error ({_summarize_error(policy_error)}); failing safe.",
            )
        self.state.escalation_decision = decision
        return "escalate" if decision.escalate else "resolve"

    @listen("escalate")
    def escalate_to_engineering(self):
        try:
            if not USE_LIVE_LLM:
                raise RuntimeError("SIMULATION_MODE active - skipping live LLM call by design.")
            agent = build_escalation_agent(get_llm())
            ticket = _run_single_agent_task(agent, build_escalation_task(agent, self.state))
            if ticket is None:
                raise ValueError("Escalation task did not return valid structured output.")
        except Exception as stage_error:
            self.state.stage_errors.append(f"escalate_to_engineering: {_summarize_error(stage_error)}")
            ticket = simulate_escalation_ticket(self.state)

        ticket.ticket_id = f"ENG-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self.state.flow_id}"
        self.state.engineering_ticket = ticket
        self.state.final_customer_message = (
            f"{self.state.troubleshooting.customer_response}\n\n"
            f"I've also escalated this to our engineering team (ticket {ticket.ticket_id}, "
            f"priority {ticket.priority}) so it gets a deeper look beyond what I can fix directly."
        )
        return "workflow_complete"

    @listen("resolve")
    def finalize_resolution(self):
        self.state.final_customer_message = self.state.troubleshooting.customer_response
        return "workflow_complete"


# ---------------------------------------------------------------------------
# 8. Runner
# ---------------------------------------------------------------------------
def _kickoff_flow_safely(flow: SupportWorkflowFlow):
    """Run flow.kickoff() in a fresh thread so asyncio.run() inside CrewAI never
    collides with an already-running event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(flow.kickoff).result()


def run_support_workflow(query: str, customer_tier: str = "standard",
                         customer_reported_urgency: str = "normal") -> SupportFlowState:
    try:
        flow = SupportWorkflowFlow()
        flow.state.query = query
        flow.state.customer_tier = customer_tier
        flow.state.customer_reported_urgency = customer_reported_urgency
        _kickoff_flow_safely(flow)
        return flow.state
    except Exception as fatal_error:
        error_state = SupportFlowState(query=query, customer_tier=customer_tier,
                                       customer_reported_urgency=customer_reported_urgency)
        error_state.stage_errors.append(f"FATAL workflow error: {_summarize_error(fatal_error)}")
        error_state.final_customer_message = (
            "We hit an internal error processing your request. This has been logged for review."
        )
        return error_state


# ---------------------------------------------------------------------------
# 9. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SaaS Product Support Multi-Agent Webhook",
    description=("External webhook front-end for the CrewAI Technical Issue Diagnosis multi-agent "
                 "workflow (diagnosis -> non-RAG knowledge reasoning -> troubleshooting -> "
                 "deterministic escalation policy -> optional engineering ticket)."),
    version="1.0.0",
)

# Tighten via ALLOWED_ORIGINS="https://a.com,https://b.com" in production.
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class WebhookRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Raw customer support message.")
    customer_tier: str = Field("standard", description="standard | pro | enterprise")
    customer_reported_urgency: str = Field("normal", description="normal | urgent")


class DiagnosisOut(BaseModel):
    issue_category: str
    symptoms_summary: str
    error_codes: List[str]
    severity: str
    urgency: str
    confidence: float


class KnowledgeOut(BaseModel):
    known_issue: bool
    kb_article_ids: List[str]
    root_cause_hypothesis: str
    applicable_policy: str
    suggested_fix_type: str
    reasoning_confidence: float


class TroubleshootingOut(BaseModel):
    customer_response: str
    steps_provided: List[str]
    resolution_status: str
    customer_action_required: bool


class EscalationOut(BaseModel):
    escalate: bool
    risk_score: float
    triggered_rules: List[str]
    rationale: str


class EngineeringTicketOut(BaseModel):
    ticket_id: str
    priority: str
    title: str
    summary_for_engineering: str
    reproduction_context: str
    affected_component: str
    customer_impact: str
    recommended_next_step: str


class WebhookResponse(BaseModel):
    flow_id: str
    customer_tier: str
    customer_reported_urgency: str
    query: str
    diagnosis: Optional[DiagnosisOut] = None
    knowledge: Optional[KnowledgeOut] = None
    troubleshooting: Optional[TroubleshootingOut] = None
    escalation_decision: Optional[EscalationOut] = None
    engineering_ticket: Optional[EngineeringTicketOut] = None
    final_customer_message: str
    stage_errors: List[str]
    simulation_mode: bool
    processing_time_ms: int


def _state_to_response(state: SupportFlowState, elapsed_ms: int) -> WebhookResponse:
    return WebhookResponse(
        flow_id=state.flow_id,
        customer_tier=state.customer_tier,
        customer_reported_urgency=state.customer_reported_urgency,
        query=state.query,
        diagnosis=DiagnosisOut(**state.diagnosis.model_dump()) if state.diagnosis else None,
        knowledge=KnowledgeOut(**state.knowledge.model_dump()) if state.knowledge else None,
        troubleshooting=TroubleshootingOut(**state.troubleshooting.model_dump()) if state.troubleshooting else None,
        escalation_decision=EscalationOut(**state.escalation_decision.model_dump()) if state.escalation_decision else None,
        engineering_ticket=EngineeringTicketOut(**state.engineering_ticket.model_dump()) if state.engineering_ticket else None,
        final_customer_message=state.final_customer_message,
        stage_errors=state.stage_errors,
        simulation_mode=SIMULATION_MODE or not CREWAI_IMPORT_OK,
        processing_time_ms=elapsed_ms,
    )


def _check_secret(x_webhook_secret: Optional[str]) -> None:
    if not WEBHOOK_SHARED_SECRET:
        return
    if x_webhook_secret != WEBHOOK_SHARED_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing X-Webhook-Secret header.")


@app.get("/health")
def health():
    # Render health check path: /health
    return {"status": "ok", "simulation_mode": SIMULATION_MODE or not CREWAI_IMPORT_OK,
            "auth_enabled": bool(WEBHOOK_SHARED_SECRET)}


@app.get("/")
def root():
    return {
        "service": "SaaS Product Support Multi-Agent Webhook",
        "endpoints": {"health": "GET /health", "webhook": "POST /webhook"},
        "docs": "/docs",
    }


@app.post("/webhook", response_model=WebhookResponse)
def webhook(payload: WebhookRequest, x_webhook_secret: Optional[str] = Header(default=None)):
    """Sync handler: FastAPI runs it in a worker thread, so it never blocks the event loop."""
    _check_secret(x_webhook_secret)

    if payload.customer_tier not in ("standard", "pro", "enterprise"):
        raise HTTPException(status_code=422, detail="customer_tier must be one of: standard, pro, enterprise")
    if payload.customer_reported_urgency not in ("normal", "urgent"):
        raise HTTPException(status_code=422, detail="customer_reported_urgency must be one of: normal, urgent")

    started = time.monotonic()
    logger.info("Webhook request (tier=%s, urgency=%s): %.80s",
                payload.customer_tier, payload.customer_reported_urgency, payload.query)

    try:
        state = run_support_workflow(
            query=payload.query,
            customer_tier=payload.customer_tier,
            customer_reported_urgency=payload.customer_reported_urgency,
        )
    except Exception as exc:
        logger.exception("Unhandled error running support workflow")
        raise HTTPException(status_code=500, detail=f"Internal workflow error: {exc.__class__.__name__}") from exc

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info("Flow %s complete in %dms (escalate=%s)", state.flow_id, elapsed_ms,
                state.escalation_decision.escalate if state.escalation_decision else None)
    return _state_to_response(state, elapsed_ms)


logger.info("Startup: SIMULATION_MODE=%s, USE_LIVE_LLM=%s, auth_enabled=%s",
            SIMULATION_MODE, USE_LIVE_LLM, bool(WEBHOOK_SHARED_SECRET))


# Local run: python main.py   (Render uses the uvicorn start command instead)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
