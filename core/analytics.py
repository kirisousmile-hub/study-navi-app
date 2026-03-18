import streamlit as st


def calc_score_percent(total: int, correct: int) -> int:
    if total <= 0:
        return 0
    return int((correct / total) * 100)


def show_weak_heatmap(load_learning_profile, get_weak_topics_sorted):
    profile = load_learning_profile()

    if not profile:
        st.info("弱点データがまだありません。")
        return

    st.subheader("🔥 弱点トピック")
    st.caption("ミスが多いトピックを上位5件まで表示します。")

    weak_topics = get_weak_topics_sorted(profile)

    for i, (_, topic) in enumerate(weak_topics[:5], start=1):
        st.write(f"{i}位 : {topic}")


def show_learning_dashboard(load_learning_profile):
    profile = load_learning_profile()

    if not profile:
        st.info("まだ理解度データがありません。")
        return

    st.subheader("📊 学習ダッシュボード")
    st.caption("各トピックの正答率を表示します。")

    topics = []
    scores = []

    for topic, data in profile.items():
        total = data.get("total", 0)
        correct = data.get("correct", 0)

        topics.append(topic[:20])
        scores.append(calc_score_percent(total, correct))

    chart_data = {
        "topic": topics,
        "score": scores,
    }

    st.bar_chart(chart_data, x="topic", y="score")