import streamlit as st
import streamlit.components.v1 as components


def render_scroll_to_top_button() -> None:
    components.html(
        """
        <div style="position: fixed; bottom: 24px; right: 24px; z-index: 99999;">
            <button
                onclick="window.parent.scrollTo({top: 0, behavior: 'smooth'});"
                style="
                    background: #4f8cff;
                    color: white;
                    border: none;
                    border-radius: 999px;
                    padding: 10px 14px;
                    font-size: 14px;
                    font-weight: bold;
                    cursor: pointer;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
                "
            >
                ↑ 上へ
            </button>
        </div>
        """,
        height=0,
    )


def render_sources_block(docs, format_source_page) -> None:
    if not docs:
        st.caption("参照なし")
        return

    st.markdown("### 参照（sources）")

    for i, d in enumerate(docs[:4], start=1):
        title = format_source_page(d.metadata)
        content = d.page_content.strip()

        summary = content[:180].replace("\n", " ")
        if len(content) > 180:
            summary += "..."

        st.markdown(f"**{i}. {title}**")
        st.caption(summary)

        with st.expander("本文を見る"):
            st.write(content[:1200])


def render_last_answer_block(last_q, last_answer, last_hits, render_sources_block_func) -> None:
    st.subheader("直前の回答")
    if last_q:
        st.caption(f"Q. {last_q}")

    left, right = st.columns([1.2, 1])

    with left:
        st.write(last_answer)

    with right:
        render_sources_block_func(last_hits)


def render_self_test_block(
    test: dict,
    llm,
    update_profile,
    add_log,
    register_weak,
    grade_answer_func,
) -> None:
    st.subheader("自己テスト")

    for i, item in enumerate(test["items"]):
        st.markdown(f"### Q{i+1}. {item['question']}")
        user_key = f"user_answer_{i}"
        user_answer = st.text_area(
            "あなたの回答",
            key=user_key
        )

        if st.button(f"採点する Q{i+1}", key=f"grade_{i}"):
            result = grade_answer_func(
                item["topic"],
                item["question"],
                user_answer,
                item["reference"],
                llm,
                update_profile,
                add_log,
                register_weak
            )
            st.session_state["self_test_results"][str(i)] = result

        saved_result = st.session_state["self_test_results"].get(str(i))
        if saved_result:
            st.markdown("### AI採点")
            st.write(saved_result)

            if item.get("reference"):
                st.markdown("### 模範回答")
                st.write(item["reference"])

            if item.get("explanation"):
                st.markdown("### 解説")
                st.write(item["explanation"])


def render_quick_drill_block(
    get_drill_topic_func,
    generate_drill_question_func,
    grade_answer_func,
    llm,
    update_profile,
    add_log,
    register_weak,
) -> None:
    st.subheader("🧠 クイック確認ドリル")
    st.caption("弱点トピックを短い問題で素早く確認する、軽めの反復モードです。")

    if st.button("クイック問題を作成"):
        topic = get_drill_topic_func()
        item = generate_drill_question_func(topic, llm)
        st.session_state["drill_item"] = item
        st.session_state["drill_result"] = None

    drill_item = st.session_state.get("drill_item")

    if drill_item:
        st.write("### クイック問題")
        st.write(drill_item["question"])

        answer = st.text_area("あなたの回答", key="drill_answer")

        if st.button("回答を採点", key="drill_grade_btn"):
            result = grade_answer_func(
                drill_item["topic"],
                drill_item["question"],
                answer,
                drill_item["reference"],
                llm,
                update_profile,
                add_log,
                register_weak
            )
            st.session_state["drill_result"] = result

        saved_result = st.session_state.get("drill_result")
        if saved_result:
            st.write("### AI採点")
            st.write(saved_result)

            if drill_item.get("reference"):
                st.markdown("### 模範回答")
                st.write(drill_item["reference"])

            if drill_item.get("explanation"):
                st.markdown("### 解説")
                st.write(drill_item["explanation"])

        if st.button("次の問題を作成", key="drill_next_btn"):
            topic = get_drill_topic_func()
            item = generate_drill_question_func(topic, llm)
            st.session_state["drill_item"] = item
            st.session_state["drill_result"] = None
            st.rerun()


def render_deep_learning_block(
    build_next_question_func,
    grade_answer_func,
    llm,
    update_profile,
    add_log,
    register_weak,
) -> None:
    st.subheader("🚀 深掘り学習モード")
    st.caption("教材をもとに問題・模範回答・解説を生成して、じっくり理解を深めます。")

    if st.button("深掘り学習を開始"):
        st.session_state["loop_question"] = build_next_question_func()
        st.session_state["loop_grade_result"] = None

    loop_data = st.session_state.get("loop_question")

    if loop_data:
        st.write("### 深掘り問題")
        st.write(loop_data["question"])

        loop_answer = st.text_area("あなたの回答", key="loop_answer")

        if st.button("回答を採点", key="loop_grade_btn"):
            result = grade_answer_func(
                loop_data["topic"],
                loop_data["question"],
                loop_answer,
                loop_data["reference"],
                llm,
                update_profile,
                add_log,
                register_weak
            )
            st.session_state["loop_grade_result"] = result

        if st.session_state.get("loop_grade_result"):
            st.write("### AI採点")
            st.write(st.session_state["loop_grade_result"])

            st.markdown("### 模範回答")
            st.write(loop_data["reference"])

            st.markdown("### 解説")
            st.write(loop_data["explanation"])

        if st.button("次の問題を作成", key="loop_next_btn"):
            st.session_state["loop_question"] = build_next_question_func()
            st.session_state["loop_grade_result"] = None
            st.rerun()


def render_mission_curriculum_block(
    load_profile,
    llm,
    generate_today_mission_func,
    recommend_next_topic_func,
    generate_ai_curriculum_func,
    find_root_weakness_func,
) -> None:
    st.divider()

    # 今日のミッション
    st.subheader("🎯 今日のミッション")

    if st.button("今日のミッションを作成"):
        mission = generate_today_mission_func(load_profile, llm)
        st.session_state["mission"] = mission

    if "mission" in st.session_state:
        st.write(st.session_state["mission"])

    st.divider()

    # AIカリキュラム
    st.markdown("### 🤖 AIカリキュラム")
    st.markdown("### 🧭 次に学ぶトピック")

    if st.button("次に学ぶ内容を提案"):
        topic = recommend_next_topic_func(
            load_learning_profile=load_profile,
            llm=llm,
            find_root_weakness=find_root_weakness_func,
        )
        st.success(topic)

    if st.button("今日の学習メニューを作成"):
        curriculum = generate_ai_curriculum_func(load_profile, llm)
        st.session_state["ai_curriculum"] = curriculum

    curriculum = st.session_state.get("ai_curriculum")
    if curriculum:
        st.write(curriculum)


def render_analytics_tab(
    load_profile,
    llm,
    show_learning_dashboard_func,
    show_weak_heatmap_func,
    get_weak_topics_sorted_func,
    show_knowledge_map_func,
    explain_weakness_func,
    get_learning_level_func,
) -> None:
    st.header("📊 分析")

    show_learning_dashboard_func(load_profile)
    show_weak_heatmap_func(load_profile, get_weak_topics_sorted_func)

    show_knowledge_map_func()

    st.subheader("🧠 弱点分析")

    if st.button("弱点の原因を分析"):
        explanation = explain_weakness_func(load_profile, llm)
        st.session_state["weak_explain"] = explanation

    explanation = st.session_state.get("weak_explain")

    if explanation:
        st.write(explanation)

    st.divider()

    st.subheader("🏆 学習レベル")

    level = get_learning_level_func(load_profile)
    st.write(level)

    st.subheader("🧑‍🏫 AI学習アドバイス")

    if st.button("学習状況を分析"):
        profile = load_profile()

        if not profile:
            st.info("まだ学習データがありません")
        else:
            summary = []

            for topic, data in profile.items():
                total = data["total"]
                correct = data["correct"]
                score = correct / total if total else 0
                summary.append(f"{topic}:{round(score*100)}%")

            prompt = f"""
Python学習コーチとして
次の学習データを分析してください。

{summary}

・現在の理解度
・弱点
・次の学習アドバイス

を短く説明してください。
"""
            advice = llm.invoke(prompt).content
            st.write(advice)


def render_review_tab(
    load_cards_func,
    save_cards_func,
    check_due_func,
    review_score_func,
    load_weak_func,
    now_func,
) -> None:
    today_str = now_func().strftime("%Y-%m-%d")

    st.header("📝 復習")

    cards = load_cards_func()
    if not cards:
        st.info("まだカードがありません。まずは学習タブで質問 → 回答 → 『📝 直前の回答をカード化』を押してください。")
        return

    only_due = st.checkbox("今日の復習だけ表示する（next_review_date <= 今日）", value=True)

    filtered = [c for c in cards if (check_due_func(c, today_str) if only_due else True)]
    st.write(f"カード数: {len(cards)} / 表示: {len(filtered)}（今日={today_str}）")

    sort_key = st.selectbox("並び順", ["next_review_dateが古い順", "作成日時が新しい順"], index=0)
    if sort_key == "next_review_dateが古い順":
        filtered.sort(key=lambda c: c.get("next_review_date") or "0000-00-00")
    else:
        filtered.sort(key=lambda c: c.get("created_at") or "", reverse=True)

    options = [
        f"{c.get('id','????')} | {c.get('next_review_date','-')} | {c.get('topic','(no topic)')[:50]}"
        for c in filtered
    ]

    if not options:
        st.info("表示できる復習カードがありません。")
        if only_due:
            st.caption("『今日の復習だけ表示する』を外すと、まだ先のカードも表示できます。")
        return

    sel = st.selectbox("カードを選択", options, index=0)

    if not sel:
        st.info("カードが選択されていません。")
        return

    sel_id = sel.split("|")[0].strip()
    card = next((c for c in cards if c.get("id") == sel_id), None)

    if not card:
        st.error("カードが見つかりませんでした（データ不整合の可能性）。")
        return

    st.subheader("問題（自分で答えてから開く）")
    st.write(card.get("question") or card.get("topic"))

    show_answer = st.checkbox("答えを表示する", value=False)
    if show_answer:
        st.subheader("答え（保存された回答）")
        st.write(card.get("answer", ""))

        st.subheader("参照（sources）")
        srcs = card.get("sources", [])
        if srcs:
            st.markdown("\n".join([f"- {s}" for s in srcs]))
        else:
            st.write("(なし)")

    st.divider()
    st.subheader("採点して次回復習日を更新")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("表面的理解（明日）", key=f"review_surface_{card['id']}"):
            review_score_func(cards, card["id"], "表面的理解")

    with col2:
        if st.button("部分理解（2日後）", key=f"review_partial_{card['id']}"):
            review_score_func(cards, card["id"], "部分理解")

    with col3:
        if st.button("概念理解（7日後）", key=f"review_concept_{card['id']}"):
            review_score_func(cards, card["id"], "概念理解")

    with col4:
        if st.button("応用理解（14日後）", key=f"review_applied_{card['id']}"):
            review_score_func(cards, card["id"], "応用理解")

    st.divider()
    st.subheader("管理")
    if st.button("このカードを削除（危険）"):
        cards2 = [c for c in cards if c.get("id") != card.get("id")]
        save_cards_func(cards2)
        st.warning("削除しました。")
        st.rerun()

    st.divider()
    st.subheader("弱点トピック")

    weak = load_weak_func()

    if weak:
        weak_sorted = sorted(weak, key=lambda x: x["count"], reverse=True)
        for w in weak_sorted[:5]:
            st.write(f"{w['topic']} （{w['count']}回ミス）")
    else:
        st.caption("まだ弱点はありません")


def render_material_tab(
    uploaded_files,
    collect_files_func,
    save_files_func,
    embed_model,
    get_vector_db_func,
    build_vectorstore_func,
    split_target_and_skipped_paths_func,
    load_documents_from_paths_func,
    split_docs_func,
    build_index_result_message_func,
    show_skipped_files_func,
    mark_indexed_paths_func,
    show_error_func,
) -> None:
    st.header("📂 教材管理")

    local_paths = collect_files_func()

    if st.button("インデックスを作成（追加）"):
        try:
            saved_paths = []
            if uploaded_files:
                saved_paths = save_files_func(uploaded_files)

            local_paths = collect_files_func()
            candidate_paths = [*saved_paths, *local_paths]

            target_paths, skipped = split_target_and_skipped_paths_func(candidate_paths)
            raw_docs = load_documents_from_paths_func(target_paths)
            chunks = split_docs_func(raw_docs)

            if len(chunks) == 0:
                st.success(
                    build_index_result_message_func(
                        saved_paths=saved_paths,
                        local_paths=local_paths,
                        target_paths=target_paths,
                        skipped=skipped,
                        chunk_count=0,
                    )
                )
                show_skipped_files_func(skipped)
            else:
                before = get_vector_db_func(embed_model)._collection.count()
                build_vectorstore_func(chunks, embed_model)
                after = get_vector_db_func(embed_model)._collection.count()

                st.info(f"Chroma count: {before} -> {after} (+{after - before})")

                mark_indexed_paths_func(target_paths)

                st.success(
                    build_index_result_message_func(
                        saved_paths=saved_paths,
                        local_paths=local_paths,
                        target_paths=target_paths,
                        skipped=skipped,
                        chunk_count=len(chunks),
                    )
                )
                show_skipped_files_func(skipped)

        except Exception as e:
            show_error_func("インデックス作成に失敗", e)


def render_weak_training_block(
    generate_weak_question_func,
    grade_answer_func,
    llm,
    llm_creative,
    retriever_db,
    embeddings,
    load_weak_func,
    update_profile,
    add_log,
    register_weak,
    retrieve_hits_func,
) -> None:
    st.divider()
    st.subheader("AIトレーニング")

    if st.button("弱点トレーニングを開始"):
        item = generate_weak_question_func(
            load_weak_func,
            llm_creative,
            retriever_db,
            embeddings,
            retrieve_hits_func,
        )

        if item:
            st.session_state["duo_item"] = item
            st.session_state["duo_result"] = None
        else:
            st.warning("まだ弱点がありません")

    duo_item = st.session_state.get("duo_item")

    if duo_item:
        st.subheader("AIトレーニング問題")
        st.write(duo_item["question"])

        duo_answer = st.text_area("あなたの回答", key="duo_answer")

        if st.button("回答を採点", key="duo_grade_btn"):
            result = grade_answer_func(
                duo_item["topic"],
                duo_item["question"],
                duo_answer,
                duo_item["reference"],
                llm,
                update_profile,
                add_log,
                register_weak
            )
            st.session_state["duo_result"] = result

        saved_result = st.session_state.get("duo_result")
        if saved_result:
            st.write("### AI採点")
            st.write(saved_result)

            if duo_item.get("reference"):
                st.markdown("### 模範回答")
                st.write(duo_item["reference"])

            if duo_item.get("explanation"):
                st.markdown("### 解説")
                st.write(duo_item["explanation"])

            if duo_item.get("source_details"):
                st.markdown("### 参照（sources）")

                for i, src in enumerate(duo_item["source_details"], start=1):
                    st.markdown(f"**{i}. {src['title']}**")
                    st.caption(f"要点: {src['summary']}")

                    with st.expander("本文を見る"):
                        st.write(src["content"])

            elif duo_item.get("sources"):
                st.markdown("### 参照（sources）")
                st.markdown("\n".join([f"- {s}" for s in duo_item["sources"]]))


def render_coach_sidebar_block(
    build_memory_block_func,
    load_wall_memory_func,
    delete_wall_fact_func,
    delete_wall_summary_func,
    add_wall_fact_func,
    save_wall_memory_func,
) -> dict:
    wall_mode = st.selectbox(
        "学習フェーズ",
        ["A: 用語理解", "B: 設計理解", "C: コード理解"],
        index=0
    )
    wall_only_textbook = st.checkbox(
        "教材だけ参照",
        value=True,
        key="wall_only_textbook"
    )
    wall_k = st.number_input(
        "壁打ち検索k",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        key="wall_k"
    )
    use_long_memory = st.checkbox(
        "長期記憶を使う",
        value=False,
        key="use_long_memory"
    )

    st.divider()
    st.subheader("🧠 覚えさせるメモ（永続）")
    st.caption("保存済みメモ（直近）")
    st.caption("壁打ち履歴は一時保存、長期記憶と復習カードはファイル保存です。")
    st.text_area(
        "saved_memory",
        build_memory_block_func(limit=10, include_facts=True, include_summaries=True),
        height=180,
        disabled=True
    )

    st.divider()
    st.subheader("🗂 保存済みメモを個別削除")

    mem_data = load_wall_memory_func()

    facts = mem_data.get("facts", [])
    summaries = mem_data.get("summaries", [])

    if facts:
        st.caption("固定メモ")
        fact_options = {
            f"{f['id']} | {f['text'][:40]}": f["id"]
            for f in facts
        }
        selected_fact_label = st.selectbox(
            "削除する固定メモ",
            options=list(fact_options.keys()),
            key="delete_fact_select"
        )

        if st.button("固定メモを削除", key="delete_fact_btn"):
            ok = delete_wall_fact_func(fact_options[selected_fact_label])
            if ok:
                st.success("固定メモを削除しました")
                st.rerun()
            else:
                st.warning("削除対象が見つかりませんでした")

    if summaries:
        st.caption("学習要約")
        summary_options = {
            f"{s['id']} | {s['text'][:40].replace(chr(10), ' ')}": s["id"]
            for s in summaries
        }
        selected_summary_label = st.selectbox(
            "削除する学習要約",
            options=list(summary_options.keys()),
            key="delete_summary_select"
        )

        if st.button("学習要約を削除", key="delete_summary_btn"):
            ok = delete_wall_summary_func(summary_options[selected_summary_label])
            if ok:
                st.success("学習要約を削除しました")
                st.rerun()
            else:
                st.warning("削除対象が見つかりませんでした")

    mem_text = st.text_input(
        "覚えてほしいこと（例：合言葉はリンゴ）",
        key="mem_text"
    )

    colm1, colm2 = st.columns(2)
    with colm1:
        if st.button("➕ メモを保存", key="save_mem_btn"):
            if mem_text.strip():
                f = add_wall_fact_func(mem_text)
                st.success(f"保存しました id={f['id']}")
                st.rerun()
            else:
                st.warning("空です")

    with colm2:
        if st.button("🗑 メモ全消し（危険）", key="clear_mem_btn"):
            save_wall_memory_func({"facts": [], "summaries": []})
            st.warning("全メモを削除しました")
            st.rerun()

    return {
        "wall_mode": wall_mode,
        "wall_only_textbook": wall_only_textbook,
        "wall_k": wall_k,
        "use_long_memory": use_long_memory,
    }


def render_coach_summary_actions_block(
    wall_key,
    wall_summary_key,
    wall_hits_key,
    summarize_wall_history_func,
    add_wall_summary_func,
    load_review_cards_func,
    save_review_cards_func,
    make_review_card_func,
    build_wall_excellent_card_func,
    llm,
    show_error_func,
) -> None:
    st.divider()
    colA, colB = st.columns(2)

    with colA:
        if st.button("🧾 壁打ちまとめを作成"):
            hist = st.session_state[wall_key]
            hits = st.session_state.get(wall_hits_key, [])

            summary_text = summarize_wall_history_func(
                hist,
                hits,
                llm
            )

            st.session_state[wall_summary_key] = summary_text
            add_wall_summary_func(summary_text)
            st.rerun()

    with colB:
        if st.button("🗑 壁打ちをリセット"):
            st.session_state[wall_key] = []
            st.session_state.pop(wall_summary_key, None)
            st.session_state.pop(wall_hits_key, None)
            st.rerun()

    summary = st.session_state.get(wall_summary_key)
    if summary:
        st.subheader("まとめ")
        st.write(summary)

        c1, c2 = st.columns(2)

        with c1:
            if st.button("📝 壁打ちまとめをカード化"):
                cards = load_review_cards_func()
                card = make_review_card_func(
                    topic="壁打ちまとめ",
                    answer=summary,
                    hits=st.session_state.get(wall_hits_key, [])[:2],
                )
                cards.append(card)
                save_review_cards_func(cards)
                st.success(f"カード化しました！ id={card['id']}")
                st.rerun()

        with c2:
            if st.button("⭐ 壁打ちから優カードを作成"):
                try:
                    excellent = build_wall_excellent_card_func(
                        summary=summary,
                        hits=st.session_state.get(wall_hits_key, []),
                        llm=llm,
                    )

                    cards = load_review_cards_func()
                    card = make_review_card_func(
                        topic=excellent["topic"],
                        question=excellent["question"],
                        answer=excellent["answer"],
                        hits=st.session_state.get(wall_hits_key, [])[:2],
                    )
                    cards.append(card)
                    save_review_cards_func(cards)
                    st.success(f"優カード化しました！ id={card['id']}")
                    st.rerun()
                except Exception as e:
                    show_error_func("優カード化に失敗しました", e)

    st.caption(f"壁打ち履歴: {len(st.session_state[wall_key])} メッセージ")


def render_coach_chat_block(
    wall_mode,
    wall_only_textbook,
    wall_k,
    use_long_memory,
    retriever_db,
    embeddings,
    llm,
    auto_summary_turn,
    wall_key,
    wall_hits_key,
    wall_summary_key,
    map_to_course_topic_func,
    retrieve_hits_func,
    rerank_docs_func,
    coach_reply_func,
    summarize_wall_history_func,
    add_wall_summary_func,
    render_sources_block_func,
    format_source_page_func,
) -> None:
    st.caption("あなたが喋る → 根拠を差し込む → 質問で掘る、の順で進めます。")

    chat_area = st.container(height=600, border=True)

    with chat_area:
        for m in st.session_state[wall_key]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

    user_msg = st.chat_input(
        "いま何を復習したい？（例：for文 / 関数 / 例外 / import / 合言葉確認）"
    )

    if user_msg:
        st.session_state[wall_key].append(
            {"role": "user", "content": user_msg}
        )

        focus = map_to_course_topic_func(user_msg)

        raw_hits = retrieve_hits_func(
            focus["adjusted_query"] or user_msg,
            retriever_db,
            embeddings,
            k=int(wall_k),
            only_textbook=wall_only_textbook
        )

        hits = rerank_docs_func(
            focus["adjusted_query"] or user_msg,
            raw_hits,
            embeddings,
            top_k=int(wall_k)
        )

        st.session_state[wall_hits_key] = hits

        assistant_msg = coach_reply_func(
            st.session_state[wall_key],
            hits,
            wall_mode,
            llm,
            use_long_memory=use_long_memory,
            focus=focus,
        )

        st.session_state[wall_key].append(
            {"role": "assistant", "content": assistant_msg}
        )

        turns = len(st.session_state[wall_key]) // 2

        if turns >= auto_summary_turn:
            summary_text = summarize_wall_history_func(
                st.session_state[wall_key],
                hits,
                llm
            )

            add_wall_summary_func(summary_text)

            st.session_state[wall_summary_key] = summary_text

            st.session_state[wall_key] = [
                {
                    "role": "assistant",
                    "content": "ここまでの壁打ち内容を自動要約して長期記憶に保存しました。続きをどうぞ。"
                }
            ]

        st.rerun()

    hits = st.session_state.get(wall_hits_key, [])
    if hits:
        render_sources_block_func(hits, format_source_page_func)


def render_rag_question_block(
    search_k,
    retriever_db,
    embeddings,
    llm,
    llm_creative,
    wall_key,
    turn_limit,
    has_main_index_func,
    answer_with_rag_func,
    generate_self_test_func,
    render_last_answer_block_func,
    render_sources_block_func,
    format_source_page_func,
    render_self_test_block_func,
    learning_deps,
    grade_answer_func,
    count_turns_func,
    show_error_func,
) -> None:
    st.divider()
    st.subheader("質問する")

    only_textbook = st.checkbox("教材（lectures_pdf）だけ検索する", value=True)
    lesson_filter = st.text_input(
        "Lessonフィルター（例: 13）",
        value=""
    )
    question = st.text_input(
        "質問（例：Chroma永続化のしくみを自分の言葉で説明したい）",
        value=""
    )

    ask_disabled = (not question.strip()) or (not has_main_index_func(retriever_db))

    if st.button("質問する", disabled=ask_disabled):
        try:
            with st.spinner("検索＆回答中."):
                ans, hits = answer_with_rag_func(
                    question=question,
                    db=retriever_db,
                    embeddings=embeddings,
                    llm=llm,
                    k=int(search_k),
                    only_textbook=only_textbook,
                    lesson_filter=lesson_filter or None,
                )

            st.session_state["last_question"] = question
            st.session_state["last_answer"] = ans
            st.session_state["last_hits"] = hits
            st.session_state["self_test"] = None
            st.session_state["self_test_results"] = {}
        except Exception as e:
            show_error_func("回答に失敗", e)

    last_q = st.session_state.get("last_question")
    last_answer = st.session_state.get("last_answer")
    last_hits = st.session_state.get("last_hits")

    if last_answer:
        render_last_answer_block_func(
            last_q,
            last_answer,
            last_hits,
            lambda docs: render_sources_block_func(docs, format_source_page_func),
        )
        st.divider()
        st.markdown("### この内容で確認する")

        if st.button("確認問題を作る", key="make_self_test_from_last_answer"):
            topic = last_q or "直前の回答"
            test = generate_self_test_func(topic, last_hits, llm_creative)
            st.session_state["self_test"] = test
            st.session_state["self_test_results"] = {}
            st.success("確認問題を作成しました。")

        test = st.session_state.get("self_test")
        if test:
            render_self_test_block_func(
                test,
                learning_deps["llm"],
                learning_deps["update_learning_profile"],
                learning_deps["add_learning_log"],
                learning_deps["register_weak_point"],
                grade_answer_func,
            )

    turns, msgs = count_turns_func(st.session_state[wall_key])
    st.caption(f"壁打ち履歴: {turns} / {turn_limit} ターン（メッセージ {msgs}件）")