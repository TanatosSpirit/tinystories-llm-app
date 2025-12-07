import streamlit as st
from inference import Narrator

# Настройка страницы
st.set_page_config(
    page_title="TinyStories Generator",
    page_icon="",
    layout="centered"
)


# Загрузка модели (кэшируется)
@st.cache_resource
def get_model():
    return Narrator(
        model_path="model.onnx",
        tokenizer_path="tokenizer.pickle"
    )

# Заголовок
st.title("TinyStories Text Generator")
st.markdown("Генерация текста с помощью decoder-only transformer")

# Инициализация модели
try:
    model = get_model()
    st.success("Модель загружена успешно")
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

# Разделитель
st.divider()
method = st.selectbox(
    "Стратегия семплирования",
    options=["Greedy", "Only Temp", "Top-k + Temp", "Top-p + Temp", ],
    index=0
)

gen_params = {}

if method == "Greedy":
    st.info("Выбор самого вероятного токена.")
    gen_params["method"] = 0

elif method == "Only Temp":
    st.info("Контроль случайности выбора следующего токена (выше = креативнее, ниже = предсказуемее).")
    temperature = st.slider("temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05)
    gen_params["method"] = 1
    gen_params["temp"] = temperature

elif method == "Top-k + Temp":
    st.info("Случайный выбор из k штук самых вероятных токенов. Температура делает выбор внутри этого набора более или менее случайным.")

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("top_k", min_value=1, max_value=100, value=30, step=1, key="top_k")
    with col2:
        temperature = st.slider("temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05, key="temp")

    gen_params["method"] = 2
    gen_params["top_k"] = top_k
    gen_params["temp"] = temperature

elif method == "Top-p + Temp":
    st.info("Выбор из минимального набора самых вероятных токенов, чья суммарная вероятность ≥ p (nucleus sampling).  \nТемпература делает выбор внутри этого набора более или менее случайным.")

    col1, col2 = st.columns(2)
    with col1:
        top_p = st.slider("top_p", min_value=0.5, max_value=1.0, value=0.92, step=0.01, key="top_p")
    with col2:
        temperature = st.slider("temperature", min_value=0.1, max_value=2.0, value=0.8, step=0.05, key="temp")

    gen_params["method"] = 3
    gen_params["top_p"] = top_p
    gen_params["temp"] = temperature

# st.write("Итоговые параметры для `generate()`:")
# st.code(gen_params, language="python")


st.divider()

# Ввод промпта
prompt = st.text_area(
    "Введите начало истории:",
    value="Once upon a time",
    height=100,
    help="Введите текст, который модель продолжит"
)

story_length = 0
def token_generator():
    for token in model.generate_text(prompt, **gen_params):
        if isinstance(token, dict):
            global story_length
            story_length = token['length']
        else:
            yield token.replace('\n', '  \n')


if st.button("Сгенерировать", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("⚠️ Пожалуйста, введите промпт")
    else:
        with st.spinner("Генерирую текст..."):
            try:
                st.divider()
                st.subheader("История:")

                st.write_stream(token_generator())

                # Дополнительная информация
                with st.expander("Детали генерации"):
                    st.write(f"**Промпт:** {prompt}")
                    st.write(f"**Длина:** {story_length} токенов")

            except Exception as e:
                st.error(f"❌ Ошибка генерации: {e}")

# Футер
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        Decoder-only Transformer trained on TinyStories
    </div>
    """,
    unsafe_allow_html=True
)