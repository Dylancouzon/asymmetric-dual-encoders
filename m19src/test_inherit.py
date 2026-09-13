from m19src import inherit


def test_build_lock_binds_required_identities():
    lock = inherit.build_lock()
    ids = lock["identities"]
    assert lock["state"] == "locked"
    assert ids["qdrant_commit"] == "5e32ea89cb5ea9a6827a68b27291d2e2251e1b1c"
    assert ids["document_vector_shape"] == [79269, 1024]
    assert ids["document_vector_dtype"] == "float16"
    assert ids["corpus_documents"] == 79269
    assert lock["protected_inputs"]["m18_confirmation_queries_or_qrels"] == "forbidden"
