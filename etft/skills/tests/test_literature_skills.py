from etft.skills.literature_skills import SearchArxivSkill, FetchAndChunkPapersSkill

def test_skills_init():
    skill1 = SearchArxivSkill()
    assert skill1.name == "search_arxiv"
    skill2 = FetchAndChunkPapersSkill()
    assert skill2.name == "fetch_and_chunk_papers"
