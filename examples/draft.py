import ibis
from datafusion import substrait as ss
from ibis_substrait.compiler.core import SubstraitCompiler
from datafusion import SessionContext

import pyarrow as pa

t = ibis.table(
    [("a", "string"), ("b", "float"), ("c", "int32"), ("d", "int64"), ("e", "int64")],
    "t",
)

expr = t.group_by(["a", "b"]).aggregate([t.c.sum().name("sum")]).select("b", "sum")

compiler = SubstraitCompiler()
proto = compiler.compile(expr)

ctx = SessionContext()
t = pa.Table.from_batches([], t.schema().to_pyarrow())
ctx.from_arrow_table(t, "t")

# Deserialize the .proto file
substrait_plan = ss.substrait.serde.deserialize_bytes(proto.SerializeToString())

# Get the query plan
df_logical_plan = ss.substrait.consumer.from_substrait_plan(ctx, substrait_plan)

print(df_logical_plan)

# # Execute it!
# results = ctx.create_dataframe_from_logical_plan(df_logical_plan)
# print(results)
