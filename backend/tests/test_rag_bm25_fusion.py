from tools.rag import bm25, fusion


def test_tokenizer_keeps_identifiers_whole_and_in_parts():
    tokens = bm25.tokenize("approvalRequired max_tool_rounds the")

    assert {"approvalrequired", "approval", "required", "max_tool_rounds", "max", "tool", "rounds"} <= set(tokens)
    assert "the" not in tokens


def test_tokenizer_indexes_chinese_as_characters_and_pairs():
    assert bm25.tokenize("向量索引") == ["向", "量", "索", "引", "向量", "量索", "索引"]
    assert bm25.tokenize("猫") == ["猫"]


def test_bm25_ranks_rare_terms_and_respects_length():
    texts = [
        "the agent uses a tool loop",
        "the agent and many other words about nothing in particular here and there across a long line of text",
        "approvalRequired means the edit is only a preview",
        "unrelated text about cooking",
    ]
    index = bm25.Bm25Index.build(texts)

    exact = index.search("approvalRequired", 3)
    assert [position for position, _ in exact] == [2]
    assert index.search("agent", 2)[0][0] == 0  # a short document beats a long one with the same term count
    assert index.search("qqqxyz", 3) == []
    assert index.search("agent", 5, allowed={1, 3})[0][0] == 1


def test_bm25_finds_chinese_words_without_spaces():
    index = bm25.Bm25Index.build(["今天天气很好适合出门", "向量索引保存所有分块", "完全无关的内容"])

    assert index.search("向量索引", 3)[0][0] == 1


def test_reciprocal_rank_fusion_rewards_agreement_and_never_adds_raw_scores():
    fused = fusion.reciprocal_rank_fusion({"bm25": [4, 1, 2], "vector": [1, 4, 3]})

    positions = [position for position, _score, _via in fused]
    assert set(positions[:2]) == {1, 4}  # found by both retrievers
    assert positions[2:] == [2, 3] or positions[2:] == [3, 2]
    by_position = {position: (score, via) for position, score, via in fused}
    assert by_position[4][1] == {"bm25": 1, "vector": 2}
    assert all(0 < score <= 1 for score, _via in by_position.values())
    # first in both lists is the best possible chunk: exactly 1.0
    assert fusion.reciprocal_rank_fusion({"a": [7], "b": [7]})[0][1] == 1.0
    # found by one of two retrievers that ran (the other found nothing): at most half
    assert fusion.reciprocal_rank_fusion({"a": [7]}, retrievers=2)[0][1] == 0.5
    assert fusion.reciprocal_rank_fusion({}) == []
