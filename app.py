import streamlit as st
from inference import Narrator

# Настройка страницы
st.set_page_config(
    page_title="TinyStories Generator",
    page_icon="📖",
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

# Ввод промпта
prompt = st.text_area(
    "Введите начало истории:",
    value="Once upon a time",
    height=100,
    help="Введите текст, который модель продолжит"
)

# Параметры генерации
# temperature = st.slider(
#     "Temperature",
#     min_value=0.1,
#     max_value=2.0,
#     value=0.8,
#     step=0.1
# )
temperature = 0

# Кнопка генерации
if st.button("Сгенерировать", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("⚠️ Пожалуйста, введите промпт")
    else:
        with st.spinner("Генерирую текст..."):
            try:
                generated_text, prompt_length = model.generate_text(prompt)

                # Вывод результата
                st.divider()
                st.subheader("Результат:")
                st.text(generated_text)

                # Дополнительная информация
                with st.expander("Детали генерации"):
                    st.write(f"**Промпт:** {prompt}")
                    st.write(f"**Длина:** {prompt_length} токенов")
                    st.write(f"**Temperature:** {temperature}")

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