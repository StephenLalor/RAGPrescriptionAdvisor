from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from treatment_checker.build_db.vec_store import load_retriever
from treatment_checker.query_routing.query_routing import get_router_chain
from treatment_checker.rag.diagnosis_chain import get_diag_chain
from treatment_checker.rag.full_chain import get_full_chain
from treatment_checker.rag.patient_history_chain import get_hist_chain
from treatment_checker.utils import read_json


def main(question):
    # Config.
    load_dotenv()
    db_cfg = read_json("treatment_checker/build_db/db_cfg.json")
    logger.add("data/logs/log.log", colorize=True, enqueue=True, mode="w")

    # Load models.
    chat_mdl = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    emb_mdl = OpenAIEmbeddings(model=db_cfg["embedding_model"]["name"])

    # Load vector stores.
    hist_ret = load_retriever(
        emb_mdl=emb_mdl,
        dir_path=db_cfg["patient_hist"]["db"]["dir"],
        name=db_cfg["patient_hist"]["db"]["name"],
        k=db_cfg["patient_hist"]["db"]["num_return"],
    )
    drugs_ret = load_retriever(
        emb_mdl=emb_mdl,
        dir_path=db_cfg["drugs"]["db"]["dir"],
        name=db_cfg["drugs"]["db"]["name"],
        k=db_cfg["drugs"]["db"]["num_return"],
    )

    # Define main chains.
    router_chain = get_router_chain(chat_mdl)
    hist_chain = get_hist_chain(chat_mdl, hist_ret)
    diag_chain = get_diag_chain(chat_mdl, drugs_ret)
    full_chain = get_full_chain(chat_mdl, hist_ret, drugs_ret)

    # Determine quote route.
    router_chain = get_router_chain(chat_mdl)
    route = router_chain.invoke({"question": question})

    # Perform query.
    if route == "full_chain":
        result = full_chain.invoke({"question": question})
    if route == "diag_chain":
        result = diag_chain.invoke({"question": question})
    if route == "hist_chain":
        result = hist_chain.invoke(question)
    print(result)
