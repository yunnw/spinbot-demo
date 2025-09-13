import os, re, json, time, html, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SPIN-Bot Final Argument Demo", layout="wide")

# ========= Azure Configuration =========
AZURE_API_KEY = st.secrets["AZURE_API_KEY"]
AZURE_ENDPOINT = st.secrets["AZURE_ENDPOINT"]
AZURE_DEPLOYMENT = st.secrets["AZURE_DEPLOYMENT"]
AZURE_API_VERSION = st.secrets["AZURE_API_VERSION"]

ArgLabel = {"1. Correct claim": 1, "2. Incorrect claim": 2,
            "3. Supportive Data/Evidence": 3, "4. Non-supportive Data/Evidence": 4,
            "5. Supportive OR valid reasoning": 5, "6. Alternative OR invalid reasoning": 6}
ReaLabel = {"1. Relational reasoning": 1, "2. Cause-effect reasoning": 2, "3. Hypothetical reasoning": 3}
ARG_TEXT2ID = {k.split(". ", 1)[-1].lower(): str(v) for k, v in ArgLabel.items()}
REA_TEXT2ID = {k.split(". ", 1)[-1].lower(): str(v) for k, v in ReaLabel.items()}

def quick_fix_json(s: str) -> str:
    s = s.strip()
    if not s: return "[]"
    s = s.replace("}{", "},{")
    if not s.startswith("["): s = "[" + s
    if not s.endswith("]"):  s = s + "]"
    return s

def parse_json_loose(s: str) -> dict:
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.I | re.M)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        s = m.group(0)
    return json.loads(s)


def labels_from_highlights(highlight_str: str, valid_ids: set[str]) -> set[str]:
    ids = set()
    if not highlight_str or highlight_str.strip() == "[]":
        return ids
    try:
        items = json.loads(highlight_str)
    except Exception:
        items = json.loads(quick_fix_json(highlight_str))
    for it in items:
        if "id" in it and str(it["id"]) in valid_ids:
            ids.add(str(it["id"]))
        t = it.get("Type", "").strip().lower()
        if t in ARG_TEXT2ID and ARG_TEXT2ID[t] in valid_ids:
            ids.add(ARG_TEXT2ID[t])
        if t in REA_TEXT2ID and REA_TEXT2ID[t] in valid_ids:
            ids.add(REA_TEXT2ID[t])
    return ids

def generate_classification_from_pred(arg_ids, rea_ids):
    claim_label, evidence_label, reasoning_label = "No claim", "No evidence", "No reasoning"
    if "1" in arg_ids: claim_label = "Correct Claim"
    elif "2" in arg_ids: claim_label = "Incorrect Claim"
    if "3" in arg_ids: evidence_label = "Supportive Data/Evidence"
    elif "4" in arg_ids: evidence_label = "Non-supportive Data/Evidence"
    if "5" in arg_ids: reasoning_label = "Supportive OR valid reasoning"
    elif "6" in arg_ids: reasoning_label = "Alternative OR invalid reasoning"
    return {"claim": claim_label, "evidence": evidence_label, "reasoning": reasoning_label}

def calling_azure_openai(query, prompt, retry=2, check_highlights=True):
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    body = {
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        "temperature": 0.4
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        if check_highlights:
            if "HIGHLIGHTS=[" not in content:
                if retry == 0:
                    return "[]", "Error: Fail to extract anything!!!"
                return calling_azure_openai(query, prompt, retry - 1, check_highlights)
            think = content.split("</think>")[0] if "</think>" in content else content.split("HIGHLIGHTS=", 1)[0]
            return content.split("HIGHLIGHTS=", 1)[-1], think
        else:
            return content, ""
    except Exception as e:
        return "[]", f"Exception occurred: {e}"


def chat_router(user_msg: str, history: list[tuple[str,str]], section_desc: str):
    convo = history_to_text(history)
    chat_query = (
        f"CURRENT_SECTION_DESC:\n{section_desc}\n\n"
        f'priorConversation = "Here is the conversation record: DIALOGUE"\n{convo}\n\n'
        f"Student's latest message:\n{user_msg}\n\n"
        "Return only JSON with keys action and bot."
    )
    content, _ = calling_azure_openai(chat_query, PROMPTS["chat_router"], check_highlights=False)
    try:
        obj = parse_json_loose(content)
    except Exception:
        obj = {"action": "CLAIM_HELP", "bot": guard_spinbot_text(content)}
    return obj.get("action","CLAIM_HELP").upper(), guard_spinbot_text(obj.get("bot",""))


def run_component_prompt(p, tokens, prompt_str):
    hl_str, think = calling_azure_openai(" ".join(tokens), prompt_str, check_highlights=True)
    return p, hl_str, think

def load_component_prompts():
    version = "v2.2"
    paths = {
        "1_claim":             f"prompts/{version}/System_prompt_for argumentation_claim1.txt",
        "2_evidence":          f"prompts/{version}/System_prompt_for argumentation_evidence2.txt",
        "3_reasoning":         f"prompts/{version}/System_prompt_for argumentation_reasoning3.txt",
        "4_reasoning_pattern": f"prompts/{version}/System_prompt_for reasoning_patterns4.txt",
        "5_feedback":          f"prompts/{version}/System_prompt_for _feedback5.txt",
        "chat_router":         f"prompts/{version}/System_prompt_for_chat_router.txt",
    }
    out = {}
    for k, p in paths.items():
        try:
            with open(p, encoding="utf8") as f:
                out[k] = f.read().strip()
        except Exception:
            out[k] = f"[MISSING PROMPT: {p}]"
    return out

PROMPTS = load_component_prompts()

FINAL_PAGE_DESC = (
    "Section: Making Scientific Arguments "
    "In the final question, students no longer have access to the simulation. "
    "Instead, students evaluate new real-world data to predict corn harvest outcomes and form a "
    "complete argument including claim, evidence, and reasoning. The question students respond to is: "
    "\"One of your classmates found actual data collected from a corn farm that was facing a rootworm "
    "infestation, just like the garden in your school. Based on this data, some of your classmates predict "
    "that if they continue adding 10 harvest spiders, it will help improve the corn "
    "harvest in Year 6. Do you agree or disagree with this prediction? Analyze the data trend "
    "and make your own prediction about the Year 6 corn harvest. Your response should include a claim, "
    "supporting data, and valid reasoning.\" "
)

# 若 chat_router 文件缺失，就先退化复用 5_feedback 的规则
if "chat_router" not in PROMPTS or PROMPTS["chat_router"].startswith("[MISSING PROMPT"):
    PROMPTS["chat_router"] = PROMPTS.get("5_feedback","")

# 把 Final page 的描述附加到两套提示词末尾
PROMPTS["5_feedback"]  = (PROMPTS.get("5_feedback","")  + "\n\n" + FINAL_PAGE_DESC).strip()
PROMPTS["chat_router"] = (PROMPTS.get("chat_router","") + "\n\n" + FINAL_PAGE_DESC + 
    "\nReturn only JSON with keys action and bot. action ∈ {EVAL, CLAIM_HELP, EVIDENCE_HELP, REASONING_HELP, NO_IDEA, REDIRECT}. "
    "Speak as Spin-Bot tutor. ≤4 sentences. No direct answers. No examples. End with a question."
).strip()

# 
def history_to_text(history, max_turns=30):
    last = history[-max_turns:]
    lines = []
    for r, t in last:
        who = "Student" if r == "user" else "Spin-Bot"
        lines.append(f"{who}: {t}")
    return "\n".join(lines)

def guard_spinbot_text(text, max_sent=4):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.M)
    sents = re.split(r'(?<=[.!?。！？])\s+', text.strip())
    text = ' '.join([x for x in sents if x][:max_sent])
    text = re.sub(r'\b(for instance|for example|e\.g\.)\b', '', text, flags=re.I)
    if re.search(r'\b(the answer is)\b', text, flags=re.I):
        text = "I can’t provide the direct answer. What claim can you make from the trend, and what data supports it?"
    return text

def spinbot_pipeline(student_text: str, history: list[tuple[str, str]]):
    tokens = student_text.split()
    arg_pred_ids, rea_pred_ids = set(), set()
    with ThreadPoolExecutor(max_workers=4) as executor:
        component_keys = ("1_claim", "2_evidence", "3_reasoning", "4_reasoning_pattern")
        futures = {}
        for p in component_keys:
            prompt_str = PROMPTS.get(p, "")
            if not prompt_str or prompt_str.startswith("[MISSING PROMPT"):
                continue  # 缺文件就跳过
            futures[executor.submit(run_component_prompt, p, tokens, prompt_str)] = p
        for fut in as_completed(futures):
            p, hl_str, _ = fut.result()
            if p in ("1_claim", "2_evidence", "3_reasoning"):
                arg_pred_ids |= labels_from_highlights(hl_str, {"1","2","3","4","5","6"})
            else:
                rea_pred_ids |= labels_from_highlights(hl_str, {"1","2","3"})
    predicted_labels = generate_classification_from_pred(arg_pred_ids, rea_pred_ids)
    convo = history_to_text(history)
    fb_query = (
        f'priorConversation = "Here is the conversation record: DIALOGUE"\n'
        f'{convo}\n\n'
        f"Student's latest message:\n{student_text}\n\n"
        f"Classification Results:\n"
        f"Claim: {predicted_labels['claim']}\n"
        f"Evidence: {predicted_labels['evidence']}\n"
        f"Reasoning: {predicted_labels['reasoning']}"
    )
    feedback_text, _ = calling_azure_openai(fb_query, PROMPTS.get("5_feedback",""), check_highlights=False)
    feedback_text = guard_spinbot_text((feedback_text or "").strip())
    return predicted_labels, feedback_text

# for display
df = pd.DataFrame({
    "Year":[1,2,3,4,5],
    "# of Corn Planted":[130]*5,
    "# of Corn Harvested":[130,97,91,84,80],
    "Harvested Spiders":[0,0,10,10,10],
    "Rootworms Eggs Initial":[0,18,29,41,41],
    "Rootworms Eggs Final":[0,53,89,89,100]
})

# format
st.markdown("""
<style>
.chatbox { position: fixed; left: 24px; bottom: 84px; width: 420px; max-height: 520px;
  background: #fff; border: 1px solid #ddd; border-radius: 14px;
  box-shadow: 0 8px 24px rgba(0,0,0,0.15); padding: 10px; z-index: 10000; overflow: auto; }
.msg-row { display: flex; gap: 8px; margin: 10px 0; align-items: flex-end; }
.msg-row.user { justify-content: flex-end; }
.msg-row.bot  { justify-content: flex-start; }
.avatar { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  background: #e5e7eb; font-size: 16px; }
.bubble { max-width: 78%; padding: 10px 12px; border-radius: 14px; line-height: 1.45; }
.bubble.user { background: #e6f0ff; border: 1px solid #bfd3ff; color: #0f172a; border-bottom-right-radius: 6px; }
.bubble.bot  { background: #f8fafc; border: 1px solid #e5e7eb; color: #0f172a; border-left: 4px solid #f59e0b; border-bottom-left-radius: 6px; }
.name { font-size: 12px; color: #6b7280; margin-bottom: 4px; }
.chat-header { font-weight: 700; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# set status
if "chat_open" not in st.session_state:    st.session_state.chat_open = True
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "cer_text" not in st.session_state:     st.session_state.cer_text = ""

# page
left, right = st.columns([1.1, 1.0])
with left:
    st.subheader("Writing a Complete Scientific Argument")
    st.write("Based on the **real-world data**, make a claim about Year 6 corn harvest and justify with data + reasoning.")
    st.dataframe(df, use_container_width=True)
    st.line_chart(df[["# of Corn Harvested","Harvested Spiders","Rootworms Eggs Initial","Rootworms Eggs Final"]])

def on_get_feedback():
    text = st.session_state.cer_text.strip()
    if not text: return
    st.session_state.chat_history.append(("user", text))
    with st.spinner("Spin-Bot is thinking…"):
        _, bot_reply = spinbot_pipeline(text, st.session_state.chat_history)
    st.session_state.chat_history.append(("assistant", bot_reply.strip().strip('“”"')))
    st.session_state.chat_open = True

with right:
    st.write("**Your Scientific Argument** (Claim + Evidence + Reasoning)")
    st.text_area("Please type your argument here.", height=180, key="cer_text")
    st.button("Get Feedback", type="primary", on_click=on_get_feedback)


if st.session_state.get("chat_open", False):
    user_msg = st.chat_input("Type your message here")
    if user_msg:
        clean_msg = user_msg.strip()
        st.session_state.chat_history.append(("user", clean_msg))
        with st.spinner("Spin-Bot is thinking…"):
            action, bot_text = chat_router(clean_msg, st.session_state.chat_history, FINAL_PAGE_DESC)
            if action == "EVAL":
                _, bot_text = spinbot_pipeline(clean_msg, st.session_state.chat_history)
        st.session_state.chat_history.append(("assistant", bot_text.strip().strip('“”"')))
        st.rerun()


def render_msg(role: str, msg: str):
    safe = html.escape(msg).replace("\n", "<br>")
    if role == "user":
        st.markdown(f"""
<div class="msg-row user">
  <div class="bubble user"><div class="name">You</div>{safe}</div>
  <div class="avatar">🧑</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
<div class="msg-row bot">
  <div class="avatar">🤖</div>
  <div class="bubble bot"><div class="name">Spin-Bot</div>{safe}</div>
</div>
""", unsafe_allow_html=True)

def render_chat():
    st.markdown('<div class="chatbox">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header">Spin-Bot Chat</div>', unsafe_allow_html=True)
    for role, msg in st.session_state.chat_history:
        render_msg(role, msg)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 96px'></div>", unsafe_allow_html=True)

if st.session_state.chat_open:
    render_chat()

if not AZURE_API_KEY:
    st.warning("AZURE_API_KEY missing.")
