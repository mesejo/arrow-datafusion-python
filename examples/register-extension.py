from pathlib import Path
import pyarrow as pa
import json

from datafusion.context import SessionContext


def register_plugin_json(context: SessionContext, path: Path, func: str):
    # access the inner context because as of now the function register_plugin is not exposed
    context.ctx.register_plugin(path, func)


ctx = SessionContext()
extension_path = (
    Path(__file__).parent.resolve() / "extension" / "libdatafusion_plugin_json.so"
)
function_to_call = "json_functions"

# Sample data
data = [
    {
        "id": 1,
        "name": "Alice",
        "json_data": json.dumps({"age": 30, "city": "New York"}),
    },
    {
        "id": 2,
        "name": "Bob",
        "json_data": json.dumps({"age": 25, "city": "San Francisco"}),
    },
    {
        "id": 3,
        "name": "Charlie",
        "json_data": json.dumps({"age": 35, "city": "London"}),
    },
]

# Define the schema
schema = pa.schema(
    [
        ("id", pa.int64()),
        ("name", pa.string()),
        ("json_data", pa.string()),  # JSON data stored as string
    ]
)

# Create arrays for each column
id_array = pa.array([row["id"] for row in data], type=pa.int64())
name_array = pa.array([row["name"] for row in data], type=pa.string())
json_array = pa.array([row["json_data"] for row in data], type=pa.string())

# Create the table
table = pa.Table.from_arrays([id_array, name_array, json_array], schema=schema)

ctx.register_record_batches("json_table", [table.to_batches()])
register_plugin_json(ctx, extension_path, function_to_call)
res = ctx.sql(
    "select name, json_get_str(json_data, 'city') from json_table"
).to_pandas()
print(res)
