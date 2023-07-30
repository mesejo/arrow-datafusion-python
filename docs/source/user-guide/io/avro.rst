Avro
====

`Avro <https://avro.apache.org/>`_ is a serialization format for record data. Reading an avro file is very straightforward
with :meth:`.SessionContext.read_avro`

.. code-block:: python


    from datafusion import SessionContext

    ctx = SessionContext()
    df = ctx.read_avro("file.avro")