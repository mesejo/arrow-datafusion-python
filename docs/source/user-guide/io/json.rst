JSON
====
`JSON <https://www.json.org/json-en.html>`_ (JavaScript Object Notation) is a lightweight data-interchange format.
When it comes to reading a JSON file, using :meth:`.SessionContext.read_json` is a simple and easy

.. code-block:: python


    from datafusion import SessionContext

    ctx = SessionContext()
    df = ctx.read_avro("file.json")