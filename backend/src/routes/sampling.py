from fastapi import APIRouter, Body
from fastapi.responses import Response

from src.controllers.dependencies import Actor, Db, Writer
from src.schemas.sampling import ClaimCreate, ReceiptCreate, SamplingPlanCreate, SupplementOpen
from src.services import sampling as service
from src.utils.csv_files import write_rows

router = APIRouter(prefix='/sampling', tags=['sampling'])


@router.get('')
def list_plans(session: Db, actor: Actor):
    return service.list_plans(session, actor)


@router.post('', status_code=201)
def create_plan(payload: SamplingPlanCreate, session: Db, actor: Writer):
    return service.create_plan(session, payload, actor)


@router.get('/{plan_id}')
def plan_detail(plan_id: int, session: Db, actor: Actor):
    return service.plan_detail(session, plan_id, actor)


@router.get('/{plan_id}/reconciliation')
def reconciliation(plan_id: int, session: Db, actor: Writer):
    # 含隐藏末检结论，仅分析员/负责人用于事后对账；领单视图始终是盲样。
    return service.reconciliation(session, plan_id, actor)


@router.post('/{plan_id}/receipts', status_code=201)
def submit_receipt(plan_id: int, payload: ReceiptCreate, session: Db, actor: Writer):
    return service.submit_receipt(session, plan_id, payload, actor)


@router.post('/{plan_id}/supplement', status_code=200)
def open_supplement(plan_id: int, session: Db, actor: Writer,
                     payload: SupplementOpen | None = Body(default=None)):
    return service.open_supplement(session, plan_id, payload or SupplementOpen(), actor)


@router.post('/{plan_id}/claims', status_code=201)
def claim_order(plan_id: int, payload: ClaimCreate, session: Db, actor: Writer):
    return service.claim_order(session, plan_id, payload, actor)


@router.get('/{plan_id}/export/{key}')
def export_table(plan_id: int, key: str, session: Db, actor: Actor):
    table = service.export_sampling(session, plan_id, key, actor)
    return Response(
        write_rows(table['columns'], table['rows']),
        media_type='text/csv',
        headers={'Content-Disposition':
                 f'attachment; filename="sampling-{plan_id}-{table["key"]}.csv"'},
    )
