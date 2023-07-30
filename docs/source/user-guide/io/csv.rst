CSV
===

Reading a csv is very straightforward with :meth:`.SessionContext.read_csv`

.. code-block:: python


    from datafusion import SessionContext

    ctx = SessionContext()
    df = ctx.read_csv("file.csv")

An alternative is to use :meth:`.SessionContext.register_csv`

.. code-block:: python

    ctx.register_csv("file", "file.csv")
    df = ctx.table("file")