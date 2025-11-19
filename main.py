import os
import re
from datetime import datetime
from typing import List, Optional

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
    r"\b(Circular|Notice|Advisory|Guideline|Directive)\s*(No\.?|Number|Ref\.?|Reference)?\s*[:#-]?\s*([A-Za-z0-9\-\/_.]+)",
    r"\bRef(?:erence)?\s*(No\.|Number)?\s*[:#-]?\s*([A-Za-z0-9\-\/_.]+)",
    r"\bNo\.?\s*[:#-]?\s*([A-Za-z0-9\-\/_.]+)",
]

DATE_PATTERNS = [
    r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    r"\b(\d{4})-(\d{2})-(\d{2})\b",
    r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b",
]


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


def detect_departments(text: str) -> List[str]:
    found = []
    for dept, patterns in DEPARTMENT_KEYWORDS.items():
        for pat in patterns:
            if re.search(pat, text, re.IGNORECASE):
                found.append(dept)
                break
    if not found:
        found = ["Compliance"]
    seen = set()
    ordered = []
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


# internal helper so we can reuse for text or file uploads

def run_analysis(text: str) -> AnalyzeResponse:
    text = text or ""
    text = re.sub(r"\x00", " ", text)

    regulator = extract_regulator(text)
    reference = extract_reference(text)
    date = extract_date(text)
    departments = detect_departments(text)
    title = generate_title(text, regulator, reference)
    bullets = make_summary(text, regulator, reference, date)
    memo = format_memo(title, regulator, reference, date, departments, bullets)

    return AnalyzeResponse(
        title=title,
        regulator=regulator,
        reference=reference,
        date=date,
        departments=departments,
        summary_bullets=bullets,
        memo=memo,
    )


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
