"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal

# Example schemas (replace with your own):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# --------------------------------------------------
# Regulatory Circular Register Schemas (used by app)
# --------------------------------------------------

class Circular(BaseModel):
    """
    Register of analyzed regulatory circulars
    Collection name: "circular"
    """
    title: str = Field(..., description="Generated title for the circular")
    regulator: Optional[str] = Field(None, description="Identified regulator")
    reference: Optional[str] = Field(None, description="Circular number or reference")
    date: Optional[str] = Field(None, description="Date string in human readable form")
    departments: List[str] = Field(default_factory=list, description="Detected target departments")
    summary_bullets: List[str] = Field(default_factory=list, description="Key points summary")
    memo: str = Field(..., description="Formatted internal memo text")
    raw_text: str = Field(..., description="Original circular text content")
    status: Literal["open", "in_progress", "closed"] = Field("open", description="Lifecycle status for the circular")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags/labels")

class CircularAssignment(BaseModel):
    """
    Per-department tracking for each circular
    Collection name: "circularassignment"
    """
    circular_id: str = Field(..., description="Associated circular document id (string)")
    department: str = Field(..., description="Department name")
    is_binding: bool = Field(True, description="Whether the circular is binding for this department")
    status: Literal["pending", "in_progress", "compliant", "non_compliant"] = Field("pending", description="Compliance status for this department")
    notes: Optional[str] = Field(None, description="Optional notes or comments")

# Note: The Flames database viewer will automatically:
# 1. Read these schemas from GET /schema endpoint
# 2. Use them for document validation when creating/editing
# 3. Handle all database operations (CRUD) directly
# 4. You don't need to create any database endpoints!
