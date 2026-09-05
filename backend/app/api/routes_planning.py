"""
POST /experiments/plan-sample-size

Pre-experiment planning — answers "how many users would I need?" BEFORE
any dataset exists, from an assumed baseline and the smallest relative
effect worth detecting. This is the inverse problem from the post-hoc
power analysis shown in a finished report (see
app/stats/power_analysis.py::compute_power_analysis, used by the
/experiments/analyze pipeline once real data is available).

Deliberately its own router/file rather than added to
routes_experiments.py: this endpoint takes no dataset_id, stores
nothing, and has no LangGraph/LLM involvement at all — pure
statsmodels math behind a thin validation layer, closer in spirit to
routes_system.py than to the analysis pipeline.
"""

import math

from fastapi import APIRouter, HTTPException

from app.core.rate_limit import rate_limit
from fastapi import Depends
from app.schemas.hypothesis import SampleSizePlanRequest, SampleSizePlanResponse
from app.stats.power_analysis import plan_required_sample_size

router = APIRouter(prefix="/experiments", tags=["planning"])


@router.post(
    "/plan-sample-size",
    response_model=SampleSizePlanResponse,
    dependencies=[Depends(rate_limit("plan_sample_size", max_requests=60))],
)
def plan_sample_size_route(request: SampleSizePlanRequest) -> SampleSizePlanResponse:
    try:
        plan = plan_required_sample_size(
            baseline_rate=request.baseline_rate,
            mde_relative_pct=request.mde_relative_pct,
            metric_type=request.metric_type,
            baseline_std=request.baseline_std,
            num_variants=request.num_variants,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    estimated_days = None
    if request.daily_traffic_per_arm:
        estimated_days = math.ceil(plan.required_n_per_arm / request.daily_traffic_per_arm)

    return SampleSizePlanResponse(plan=plan, estimated_days=estimated_days)
