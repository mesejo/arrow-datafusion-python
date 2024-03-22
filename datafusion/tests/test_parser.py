import ibis

from datafusion import parser, ContextProvider


def test_simple_plan():
    t = ibis.table(
        [
            ("a", "string"),
            ("b", "float"),
            ("c", "int32"),
            ("d", "int64"),
            ("e", "int64"),
        ],
        "t",
    )
    plan = parser.parse_sql(
        "select * from t", ContextProvider({"t": t.schema().to_pyarrow()})
    )
    assert plan is not None


def test_plan_with_filter():
    t = ibis.table(
        [
            ("a", "string"),
            ("b", "float"),
            ("c", "int32"),
            ("d", "int64"),
            ("e", "int64"),
        ],
        "t",
    )
    plan = parser.parse_sql(
        "select t.a, b from t where t.c > 4",
        ContextProvider({"t": t.schema().to_pyarrow()}),
    )
    assert plan is not None


def test_plan_with_group_by():
    t = ibis.table(
        [
            ("a", "string"),
            ("b", "float"),
            ("c", "int32"),
            ("d", "int64"),
            ("e", "int64"),
        ],
        "t",
    )
    sql = """SELECT
              t0.b,
              t0.sum
            FROM (
              SELECT
                t1.a AS a,
                t1.b AS b,
                SUM(t1.c) AS sum
              FROM t AS t1
              GROUP BY
                1,
                2
            ) AS t0
    """
    plan = parser.parse_sql(sql, ContextProvider({"t": t.schema().to_pyarrow()}))
    assert plan is not None
