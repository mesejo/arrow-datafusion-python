SQL
===

DataFusion also offers a SQL API, read the full reference `here <https://arrow.apache.org/datafusion/user-guide/sql/index.html>`_

.. ipython:: python

    import datafusion
    from datafusion import col
    import pyarrow

    # create a context
    ctx = datafusion.SessionContext()

    # register a CSV
    ctx.register_csv('pokemon', 'pokemon.csv')

    # create a new statement via SQL
    df = ctx.sql('SELECT "Attack"+"Defense", "Attack"-"Defense" FROM pokemon')

    # collect and convert to pandas DataFrame
    df.to_pandas()