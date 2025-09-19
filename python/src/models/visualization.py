from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChartConfig(BaseModel):
    chart_type: str  # bar, line, scatter, heatmap, etc.
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    data_source: str
    styling: Dict[str, Any] = Field(default_factory=dict)
    interactive: bool = True


class Visualization(BaseModel):
    viz_id: str
    config: ChartConfig
    file_path: Optional[str] = None
    s3_location: Optional[str] = None
    thumbnail_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    size_bytes: Optional[int] = None


class Dashboard(BaseModel):
    dashboard_id: str
    title: str
    description: Optional[str] = None
    visualizations: List[Visualization]
    layout: Dict[str, Any] = Field(default_factory=dict)
    export_formats: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ReportSection(BaseModel):
    section_id: str
    title: str
    content: str
    visualizations: List[str] = Field(default_factory=list)  # viz_ids
    order: int = 0


class ResearchReport(BaseModel):
    report_id: str
    title: str
    abstract: str
    sections: List[ReportSection]
    dashboard: Optional[Dashboard] = None
    export_formats: List[str] = Field(default_factory=list)
    file_paths: Dict[str, str] = Field(default_factory=dict)  # format -> path
    created_at: datetime = Field(default_factory=datetime.now)


class VisualizationResults(BaseModel):
    visualizations: List[Visualization] = Field(default_factory=list)
    dashboard: Optional[Dashboard] = None
    report: Optional[ResearchReport] = None
    summary_stats: Dict[str, Any] = Field(default_factory=dict)
    export_locations: Dict[str, str] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }