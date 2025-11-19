import os
import re
from datetime import datetime
from typing import List, Optional, Literal, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="Regulatory Circular AI Tester API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Raw text of the regulatory circular")


class AnalyzeResponse(BaseModel):
    title: str
    regulator: Optional[str] = None
    reference: Optional[str] = None
    date: Optional[str] = None
    departments: List[str]
    summary_bullets: List[str]
    memo: str
    circular_id: Optional[str] = Field(None, description="Database id of the saved circular")


class AssignmentUpdate(BaseModel):
    circular_id: str
    department: str
    is_binding: Optional[bool] = None
    status: Optional[Literal["pending", "in_progress", "compliant", "non_compliant"]] = None
    notes: Optional[str] = None


# Default department keywords (used for semantic mapping)
DEPARTMENT_KEYWORDS = {
    "Compliance": [
        r"compliance", r"policy", r"governance", r"reporting", r"regulatory filing",
        r"audit", r"monitoring", r"control function", r"internal control"
    ],
    "AML / CFT / Sanctions": [
        r"aml", r"anti[-\s]?money", r"money\s?launder", r"kyc", r"cdd", r"customer due diligence",
        r"sanction", r"counter[-\s]?terrorist", r"cft", r"pep", r"ubo", r"adverse media"
    ],
    "Risk Management": [
        r"risk", r"risk appetite", r"credit risk", r"market risk", r"operational risk", r"liquidity",
        r"stress test", r"icaap", r"ifr[sS]?9"
    ],
    "Legal": [
        r"legal", r"statutory", r"enactment", r"legislation", r"contract", r"liability", r"litigation"
    ],
    "IT / Cybersecurity": [
        r"cyber", r"information security", r"infosec", r"it", r"technology", r"system", r"patch",
        r"vulnerability", r"encryption", r"access control", r"iso 27001", r"incident response",
        r"disaster recovery", r"bcp", r"drp", r"endpoint", r"malware", r"ransomware"
    ],
    "Operations / Branch Network": [
        r"branch", r"operations", r"operational", r"teller", r"processing", r"back\s?office", r"reconciliation",
        r"cash", r"clearing", r"settlement"
    ],
    "Cards & Digital Channels": [
        r"card", r"debit", r"credit card", r"payment gateway", r"pos", r"digital", r"online banking",
        r"mobile", r"internet banking", r"upi", r"wallet", r"tokenization"
    ],
    "Treasury / Finance": [
        r"treasury", r"finance", r"accounting", r"capital", r"liquidity", r"hedging", r"fx", r"funding",
        r"capital adequacy", r"basel"
    ],
    "HR & Training": [
        r"training", r"hr", r"human resources", r"awareness", r"staff", r"competency", r"fit and proper"
    ],
}

DEFAULT_DEPARTMENTS = list(DEPARTMENT_KEYWORDS.keys())

REGULATOR_PATTERNS = [
    r"(financial conduct authority|fca)",
    r"(monetary authority of singapore|mas)",
    r"(reserve bank of india|rbi)",
    r"(securities and exchange board of india|sebi)",
    r"(bangko sentral ng pilipinas|bsp)",
    r"(central bank of [a-zA-Z ]+)",
    r"(central bank)",
    r"(european central bank|ecb)",
    r"(fatf|financial action task force)",
    r"(fincen)",
    r"(new york dfs|department of financial services)",
    r"(dfsa|dubai financial services authority)",
    r"(cb|cbi|central bank of ireland)",
]

REFERENCE_PATTERNS = [
    r"\b(Circular|Notice|Advisory|Guideline|Directive)\s*(No\.?|Number|Ref\.?|Reference)?\s*[:#-]?\s*([A-Za-z0-9\-\/_ .]+)",
    r"\bRef(?:erence)?\s*(No\.|Number)?\s*[:#-]?\s*([A-Za-z0-9\-\/_ .]+)",
    r"\bNo\.?\s*[:#-]?\s*([A-Za-z0-9\-\/_ .]+)",
]

DATE_PATTERNS = [
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    r"\b(\d{4})-(\d{2})-(\d{2})\b",
    r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b",
]

# --- Department storage helpers (DB if available, else in-memory fallback) ---
_configured_departments: Optional[List[str]] = None


def _db_available():
    try:
        from database import db  # type: ignore
        return db is not None
    except Exception:
        return False


def get_configured_departments() -> List[str]:
    global _configured_departments
    # Try DB-backed configuration first
    try:
        from database import db  # type: ignore
        if db is not None:
            names = [d.get("name") for d in db["department"].find({}, {"name": 1, "_id": 0})]
            names = [n for n in names if isinstance(n, str) and n.strip()]
            if not names:
                # seed defaults
                for n in DEFAULT_DEPARTMENTS:
                    db["department"].update_one({"name": n}, {"$set": {"name": n}}, upsert=True)
                names = DEFAULT_DEPARTMENTS.copy()
            return names
    except Exception:
        pass

    # Fallback to in-memory during this process (session persistence only)
    if _configured_departments is None:
        _configured_departments = DEFAULT_DEPARTMENTS.copy()
    return _configured_departments


def add_department(name: str) -> List[str]:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Department name is required")

    try:
        from database import db  # type: ignore
        if db is not None:
            exists = db["department"].find_one({"name": name})
            if not exists:
                db["department"].insert_one({"name": name, "created_at": datetime.utcnow()})
            return get_configured_departments()
    except Exception:
        pass

    # fallback in-memory
    global _configured_departments
    if _configured_departments is None:
        _configured_departments = DEFAULT_DEPARTMENTS.copy()
    if name not in _configured_departments:
        _configured_departments.append(name)
    return _configured_departments


def delete_department(name: str) -> List[str]:
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Department name is required")

    try:
        from database import db  # type: ignore
        if db is not None:
            db["department"].delete_many({"name": name})
            return get_configured_departments()
    except Exception:
        pass

    global _configured_departments
    if _configured_departments is None:
        _configured_departments = DEFAULT_DEPARTMENTS.copy()
    _configured_departments = [d for d in _configured_departments if d != name]
    return _configured_departments


# --- Extractors ---

def extract_first(text: str, patterns: List[str], flags=re.IGNORECASE) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, flags)
        if m:
            return m.group(0)
    return None


def extract_regulator(text: str) -> Optional[str]:
    reg = extract_first(text, REGULATOR_PATTERNS)
    if reg:
        return reg.strip().rstrip(':.')
    m = re.search(r"(?:by|from) the ([A-Z][A-Za-z &(\)]+?)(?:,|\.|\n)", text)
    if m:
        return m.group(1)
    return None


def extract_reference(text: str) -> Optional[str]:
    for pat in REFERENCE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = m.group(len(m.groups())) if m.groups() else m.group(0)
            return val.strip().rstrip('.')
    return None


def extract_date(text: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                if len(m.groups()) == 3 and m.group(2).isalpha():
                    day, month, year = int(m.group(1)), m.group(2), int(m.group(3))
                    dt = datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
                elif pat.startswith("\\b(\\d{4})-"):
                    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    dt = datetime(year, month, day)
                else:
                    day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    if year < 100:
                        year += 2000
                    dt = datetime(year, month, day)
                return dt.strftime("%d %B %Y")
            except Exception:
                return m.group(0)
    return None


# --- Department detection using configured list ---

def _name_token_patterns(name: str) -> List[str]:
    tokens = re.split(r"[^A-Za-z0-9]+", name.lower())
    tokens = [t for t in tokens if len(t) > 2]
    if not tokens:
        return []
    return [rf"\b{re.escape(t)}\b" for t in tokens]


def detect_departments(text: str, choose_from: Optional[List[str]] = None) -> List[str]:
    choose_from = choose_from or get_configured_departments()
    found: List[str] = []
    for dept in choose_from:
        pats = DEPARTMENT_KEYWORDS.get(dept, None)
        if not pats:
            pats = _name_token_patterns(dept)
        for pat in pats:
            if re.search(pat, text, re.IGNORECASE):
                found.append(dept)
                break
    if not found and "Compliance" in choose_from:
        found = ["Compliance"]
    # Unique preserve order
    seen = set()
    ordered: List[str] = []
    for d in found:
        if d not in seen:
            ordered.append(d)
            seen.add(d)
    return ordered


def generate_title(text: str, regulator: Optional[str], reference: Optional[str]) -> str:
    if reference and regulator:
        return f"{regulator}: {reference}"
    if reference:
        return f"Circular {reference}"
    if regulator:
        theme = None
        themes = {
            "KYC/AML": r"kyc|aml|sanction|cft|pep|beneficial",
            "Cybersecurity": r"cyber|information security|incident|vulnerability|patch|ransom",
            "Risk Management": r"risk|stress|capital|icaap|basel",
        }
        for k, pat in themes.items():
            if re.search(pat, text, re.IGNORECASE):
                theme = k
                break
        return f"{regulator}: {theme or 'Regulatory Update'}"
    first_line = re.split(r"[\n\.]", text.strip())
    if first_line and first_line[0]:
        base = first_line[0]
        base = base[:90] + "…" if len(base) > 90 else base
        return base
    return "Regulatory Circular"


def make_summary(text: str, regulator: Optional[str], reference: Optional[str], date: Optional[str]) -> List[str]:
    bullets: List[str] = []
    if regulator:
        bullets.append(f"Regulator: {regulator}")
    if reference:
        bullets.append(f"Reference: {reference}")
    if date:
        bullets.append(f"Date: {date}")

    sentences = re.split(r"(?<=[\.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    ranked = sorted(
        sentences,
        key=lambda s: (
            -int(bool(re.search(r"must|shall|require|deadline|implement|effective|no later than|comply|prohibit", s, re.IGNORECASE))),
            -len(re.findall(r"\b\w+\b", s))
        )
    )
    for s in ranked[:7]:
        cleaned = s
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = cleaned[:220] + "…" if len(cleaned) > 220 else cleaned
        bullets.append(cleaned)
        if len(bullets) >= 10:
            break

    while len(bullets) < 5:
        bullets.append("This circular introduces requirements and expectations for the institution.")

    return bullets[:10]


def format_memo(title: str, regulator: Optional[str], reference: Optional[str], date: Optional[str], departments: List[str], bullets: List[str]) -> str:
    addressed = ", ".join(departments)
    header = []
    header.append("To: " + addressed)
    header.append("Cc: Head of Compliance, Relevant Stakeholders")
    header.append("Subject: " + title)
    if date:
        header.append(f"Date: {date}")

    regline = []
    if regulator:
        regline.append(f"Regulator: {regulator}")
    if reference:
        regline.append(f"Reference: {reference}")

    intro = "This memorandum summarizes a recently issued regulatory circular and sets out the actions required from your department(s)."

    obligations = "Key points and obligations:"\
        + "\n" + "\n".join([f"- {b}" for b in bullets])

    deadlines = "If specific deadlines are stated above, please ensure the timelines are met. Where dates are not explicit, please proceed on an urgent basis and target completion within the standard regulatory turnaround times."

    actions = (
        "Required actions:"\
        "\n- Review the above requirements and assess impact on your processes, systems and controls."
        "\n- Nominate a responsible officer and provide an implementation plan (milestones, owners, target dates)."
        "\n- Confirm any technology or resource needs."
        "\n- Provide feedback to Compliance on feasibility, risks, and timeline within 5 working days."
    )

    closing = (
        "Please send your response and ongoing updates to Compliance. "
        "Compliance will coordinate with Risk and Internal Audit as needed. "
        "Do not hesitate to contact Compliance for clarifications."
    )

    parts = [
        "\n".join(header),
        "",
        "; ".join(regline) if regline else "",
        intro,
        obligations,
        deadlines,
        actions,
        closing,
        "Regards,\nCompliance Department"
    ]

    return "\n\n".join([p for p in parts if p.strip()])


# --- Persistence helpers ---

def save_circular_to_db(analysis: AnalyzeResponse, raw_text: str) -> Optional[str]:
    """Save circular to register and seed department assignments."""
    try:
        from database import create_document, db
        if db is None:
            return None

        circular_doc: Dict[str, Any] = {
            "title": analysis.title,
            "regulator": analysis.regulator,
            "reference": analysis.reference,
            "date": analysis.date,
            "departments": analysis.departments,
            "summary_bullets": analysis.summary_bullets,
            "memo": analysis.memo,
            "raw_text": raw_text,
            "status": "open",
            "tags": [],
        }
        circular_id = create_document("circular", circular_doc)
        # Create per-department assignment docs (default binding & pending)
        for d in analysis.departments:
            db["circularassignment"].insert_one({
                "circular_id": circular_id,
                "department": d,
                "is_binding": True,
                "status": "pending",
                "notes": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            })
        return circular_id
    except Exception:
        return None


# internal helper so we can reuse for text or file uploads

def run_analysis(text: str) -> AnalyzeResponse:
    text = text or ""
    text = re.sub(r"\x00", " ", text)

    regulator = extract_regulator(text)
    reference = extract_reference(text)
    date = extract_date(text)
    dept_list = get_configured_departments()
    departments = detect_departments(text, dept_list)
    title = generate_title(text, regulator, reference)
    bullets = make_summary(text, regulator, reference, date)
    memo = format_memo(title, regulator, reference, date, departments, bullets)

    analysis = AnalyzeResponse(
        title=title,
        regulator=regulator,
        reference=reference,
        date=date,
        departments=departments,
        summary_bullets=bullets,
        memo=memo,
    )

    # Persist to register (if DB configured)
    cid = save_circular_to_db(analysis, text)
    if cid:
        analysis.circular_id = cid

    return analysis


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return run_analysis(req.text)


@app.post("/analyze-file", response_model=AnalyzeResponse)
async def analyze_file(file: UploadFile = File(...)) -> AnalyzeResponse:
    # Validate basic content types
    ct = file.content_type or ""
    name = file.filename or "uploaded"
    ext = os.path.splitext(name)[1].lower()

    try:
        raw = await file.read()
        if len(raw) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        text_content = None
        if ext == ".pdf" or ct in {"application/pdf"}:
            try:
                from PyPDF2 import PdfReader
                import io
                reader = PdfReader(io.BytesIO(raw))
                pieces = []
                for page in reader.pages:
                    pieces.append(page.extract_text() or "")
                text_content = "\n".join(pieces)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read PDF: {str(e)[:120]}")
        elif ext == ".docx" or ct in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}:
            try:
                from docx import Document
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
                    tmp.write(raw)
                    tmp.flush()
                    doc = Document(tmp.name)
                text_content = "\n".join(p.text for p in doc.paragraphs)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read DOCX: {str(e)[:120]}")
        else:
            # Fallback: try to decode as text
            try:
                text_content = raw.decode("utf-8", errors="ignore")
            except Exception:
                raise HTTPException(status_code=415, detail="Unsupported file type. Please upload PDF or DOCX.")

        text_content = (text_content or "").strip()
        if not text_content:
            raise HTTPException(status_code=400, detail="No text could be extracted from the file.")

        return run_analysis(text_content)
    finally:
        await file.close()


# --- Settings Endpoints ---
class DepartmentIn(BaseModel):
    name: str


@app.get("/settings/departments", response_model=List[str])
def list_departments() -> List[str]:
    return get_configured_departments()


@app.post("/settings/departments", response_model=List[str])
def create_department(dep: DepartmentIn) -> List[str]:
    return add_department(dep.name)


@app.delete("/settings/departments/{name}", response_model=List[str])
def remove_department(name: str) -> List[str]:
    return delete_department(name)


# --- Register & History Endpoints ---
@app.get("/history")
def list_history() -> List[Dict[str, Any]]:
    """List saved circulars for history/register page."""
    try:
        from database import db
        if db is None:
            return []
        items = []
        for d in db["circular"].find({}, {"title": 1, "departments": 1, "created_at": 1} ).sort("created_at", -1).limit(100):
            # created_at added by helper at insert time
            items.append({
                "id": str(d.get("_id")),
                "title": d.get("title"),
                "departments": d.get("departments", []),
                "created_at": d.get("created_at"),
            })
        return items
    except Exception:
        return []


@app.get("/history/{circular_id}")
def get_history_detail(circular_id: str) -> Dict[str, Any]:
    try:
        from database import db
        from bson import ObjectId
        if db is None:
            raise HTTPException(status_code=503, detail="Database not available")
        doc = db["circular"].find_one({"_id": ObjectId(circular_id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Circular not found")
        assignments = list(db["circularassignment"].find({"circular_id": str(doc.get("_id"))}))
        # Serialize
        detail = {k: v for k, v in doc.items() if k != "_id"}
        detail["id"] = str(doc.get("_id"))
        for a in assignments:
            a["id"] = str(a.get("_id"))
            a.pop("_id", None)
        detail["assignments"] = assignments
        return detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:120])


@app.post("/assignments/update")
def update_assignment(payload: AssignmentUpdate) -> Dict[str, Any]:
    """Update binding flag or compliance status/notes for a department assignment."""
    try:
        from database import db
        if db is None:
            raise HTTPException(status_code=503, detail="Database not available")
        # find assignment by circular_id + department
        q = {"circular_id": payload.circular_id, "department": payload.department}
        update: Dict[str, Any] = {"updated_at": datetime.utcnow()}
        if payload.is_binding is not None:
            update["is_binding"] = bool(payload.is_binding)
        if payload.status is not None:
            update["status"] = payload.status
        if payload.notes is not None:
            update["notes"] = payload.notes
        res = db["circularassignment"].update_one(q, {"$set": update}, upsert=True)
        ok = res.matched_count > 0 or res.upserted_id is not None
        return {"ok": ok}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)[:120])


@app.get("/")
def read_root():
    return {"message": "Regulatory Circular AI Tester API running"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
