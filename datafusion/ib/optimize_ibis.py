import ibis

from datafusion.optimizer import optimize_plan
from datafusion import parser, ContextProvider, OptimizerContext
from datafusion.ib.translate import plan_to_ibis


def optimize_sql(sql: str, catalog: dict, dialect: str = None) -> ibis.Expr:
    plan = parser.parse_sql(
        sql, ContextProvider({k: v.to_pyarrow() for k, v in catalog.items()})
    )
    optimized_plan = optimize_plan(plan, OptimizerContext())
    return plan_to_ibis(optimized_plan, catalog)
