import streamlit as st


def show_weak_heatmap(load_learning_profile, get_weak_topics_sorted):
    profile = load_learning_profile()

    if not profile:
        st.info("弱点データがまだありません")
        return

    st.subheader("🔥 弱点ヒートマップ")

    weak_topics = get_weak_topics_sorted(profile)

    for i, (_, topic) in enumerate(weak_topics[:5]):
        st.write(f"{i+1}位 : {topic}")


def show_learning_dashboard(load_learning_profile):
    profile = load_learning_profile()

    if not profile:
        st.info("まだ理解度データがありません")
        return

    st.subheader("📊 学習ダッシュボード")

    topics = []
    scores = []

    for topic, data in profile.items():
        total = data["total"]
        correct = data["correct"]

        if total == 0:
            score = 0
        else:
            score = int((correct / total) * 100)

        topics.append(topic[:20])
        scores.append(score)

    chart_data = {
        "topic": topics,
        "score": scores
    }

    st.bar_chart(chart_data, x="topic", y="score")