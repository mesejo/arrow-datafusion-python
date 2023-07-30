Parquet
=======

It is quite simple to read a parquet file using the :meth:`.SessionContext.read_parquet` function.

.. code-block:: python


    from datafusion import SessionContext

    ctx = SessionContext()
    df = ctx.read_parquet("file.parquet")

An alternative is to use :meth:`.SessionContext.register_parquet`

.. code-block:: python

    ctx.register_parquet("file", "file.parquet")
    df = ctx.table("file")