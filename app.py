import streamlit as st 
import numpy as np 
from PIL import Image
import tensorflow as tf
import json
import os
import glob

# carregando o modelo salvo
model = tf.keras.models.load_model("modelos_salvos/cnn_proprio.keras")

# definindo os nomes das classes (0 = normal, 1 = defeituoso)
CLASSES = ['normal', 'defeituoso']

# ajustando o tamanho da entrada do modelo
INPUT_SIZE = (64,64)

# caminho padrão do dataset2
DATASET2_DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset nn project u2", "dataset_2")

def preprocessar(imagem: Image.Image) -> np.ndarray:
    img = imagem.convert("RGB")
    img = img.resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32)

    # Caso os 3 canais forem diferentes (imagem colorida como dataset_2), converte para grayscale e replica em 3 canais (igual ao dataset_1)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    if not (np.allclose(r, g, atol=5) and np.allclose(g, b, atol=5)):
        gray = 0.299*r + 0.587*g + 0.114*b  # conversão padrão RGB > gray
        arr = np.stack([gray, gray, gray], axis=-1)

    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# THRESHOLD: só classifica como defeituoso se a confiança for >= 90%
THRESHOLD_DEFEITUOSO = 0.90

def prever(imagem: Image.Image):
    entrada = preprocessar(imagem)
    y_pred_prob = model.predict(entrada, verbose=0)[0]

    if len(y_pred_prob) == 1:
        prob_positivo = float(y_pred_prob[0])
        probabilidades = [1.0 - prob_positivo, prob_positivo]
    else:
        probabilidades = y_pred_prob.tolist()

    # aplicando threshold antes de definir a classe
    prob_defeituoso = probabilidades[1]
    if prob_defeituoso >= THRESHOLD_DEFEITUOSO:
        classe_predita = 1
    else:
        classe_predita = 0

    confianca = probabilidades[classe_predita]
    return CLASSES[classe_predita], confianca, probabilidades

def processar_imagem_com_json(image_pill, json_path):
    """Processa uma imagem com seu arquivo JSON de anotações e retorna estatísticas."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    instances = data.get("instances", [])
    total_modulos = 0
    defeituosos = 0
    normais = 0
    
    for inst in instances:
        corners = inst.get("corners", [])
        if not corners:
            continue
            
        xs = [c["x"] for c in corners]
        ys = [c["y"] for c in corners]
        
        x_min = max(0, int(min(xs)))
        y_min = max(0, int(min(ys)))
        x_max = min(image_pill.width,  int(max(xs)))
        y_max = min(image_pill.height, int(max(ys)))

        # validação mais restrita do bounding box
        largura = x_max - x_min
        altura  = y_max - y_min

        if x_max <= x_min or y_max <= y_min or largura < 15 or altura < 15:
            continue

        # descartando crops com proporção muito distorcida
        proporcao = largura / altura if altura > 0 else 0
        if proporcao < 0.2 or proporcao > 5.0:
            continue
            
        crop_img = image_pill.crop((x_min, y_min, x_max, y_max))
        classe, confianca, _ = prever(crop_img)
        
        total_modulos += 1
        if classe == 'defeituoso':
            defeituosos += 1
        else:
            normais += 1
    
    return total_modulos, normais, defeituosos

# Interface do Streamlit
st.set_page_config(page_title="Classificação de Imagens de Placas Fotovoltaicas", 
layout="wide", page_icon="⚡")

st.title("Classificação de Imagens de Placas Fotovoltaicas")
st.write("Modelo treinado com Deep Learning para detecção de falhas em painéis solares")
st.write("Faça upload de uma imagem para iniciar a classificação")

# Seleção da fonte da imagem: Arquivo, Câmera ou Dataset Local
st.divider()
fonte = st.radio("Selecione a fonte da imagem", options=["Arquivo", "Câmera", "Dataset Local"], horizontal=True)

image_pill: Image.Image | None = None
arquivo_json = None

if fonte == "Arquivo":
    arquivo = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png", "webp"])
    if arquivo is not None:
        image_pill = Image.open(arquivo)
    
    arquivo_json = st.file_uploader("Escolha o arquivo JSON de anotações (opcional, para imagens com múltiplas placas)", type=["json"])

elif fonte == "Câmera":
    foto = st.camera_input("Tire uma foto")
    if foto is not None:
        image_pill = Image.open(foto)

elif fonte == "Dataset Local":
    st.subheader("📂 Teste com Dataset Local")
    st.write("Selecione a pasta de um dataset local para processar todas as imagens automaticamente.")
    
    dataset_path = st.text_input(
        "Caminho do dataset (deve conter as pastas `images/` e `annotations/`):",
        value=DATASET2_DEFAULT_PATH
    )
    
    if st.button("🚀 Processar Dataset", type="primary"):
        images_dir = os.path.join(dataset_path, "images")
        annotations_dir = os.path.join(dataset_path, "annotations")
        
        if not os.path.isdir(images_dir):
            st.error(f"Pasta de imagens não encontrada: {images_dir}")
        elif not os.path.isdir(annotations_dir):
            st.error(f"Pasta de anotações não encontrada: {annotations_dir}")
        else:
            # Listar todas as imagens
            image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
            if not image_files:
                image_files = sorted(glob.glob(os.path.join(images_dir, "*.png")))
            
            if not image_files:
                st.warning("Nenhuma imagem encontrada na pasta.")
            else:
                st.info(f"Encontradas {len(image_files)} imagens para processar.")
                
                # Processando cada imagem
                resultados = []
                total_geral_modulos = 0
                total_geral_normais = 0
                total_geral_defeituosos = 0
                
                progress_bar = st.progress(0, text="Processando imagens...")
                
                for i, img_path in enumerate(image_files):
                    img_name = os.path.basename(img_path)
                    img_base = os.path.splitext(img_name)[0]
                    json_path = os.path.join(annotations_dir, f"{img_base}.json")
                    
                    if os.path.exists(json_path):
                        try:
                            img = Image.open(img_path)
                            total_mod, norm, defeit = processar_imagem_com_json(img, json_path)
                            
                            resultados.append({
                                "Imagem": img_name,
                                "Total Módulos": total_mod,
                                "✅ Normais": norm,
                                "⚠️ Defeituosos": defeit
                            })
                            
                            total_geral_modulos += total_mod
                            total_geral_normais += norm
                            total_geral_defeituosos += defeit
                        except Exception as e:
                            resultados.append({
                                "Imagem": img_name,
                                "Total Módulos": 0,
                                "✅ Normais": 0,
                                "⚠️ Defeituosos": 0
                            })
                    else:
                        # Sem JSON, classificar a imagem inteira
                        try:
                            img = Image.open(img_path)
                            classe, confianca, _ = prever(img)
                            defeit_count = 1 if classe == 'defeituoso' else 0
                            norm_count = 1 if classe == 'normal' else 0
                            
                            resultados.append({
                                "Imagem": img_name,
                                "Total Módulos": 1,
                                "✅ Normais": norm_count,
                                "⚠️ Defeituosos": defeit_count
                            })
                            
                            total_geral_modulos += 1
                            total_geral_normais += norm_count
                            total_geral_defeituosos += defeit_count
                        except Exception as e:
                            pass
                    
                    progress_bar.progress((i + 1) / len(image_files), text=f"Processando {i + 1}/{len(image_files)}...")
                
                progress_bar.progress(1.0, text="✅ Processamento concluído!")
                
                # Exibir resumo geral
                st.divider()
                st.subheader("📊 Resumo Geral")
                
                c1, c2, c3 = st.columns(3)
                c1.metric(label="Total de Módulos", value=total_geral_modulos)
                c2.metric(label="✅ Normais", value=total_geral_normais)
                c3.metric(label="⚠️ Defeituosos", value=total_geral_defeituosos)
                
                if total_geral_defeituosos > 0:
                    st.error(f"Atenção: {total_geral_defeituosos} módulo(s) defeituoso(s) detectado(s) no total!")
                elif total_geral_modulos > 0:
                    st.success("Todos os módulos estão normais!")
                
                # Exibir tabela de resultados
                st.divider()
                st.subheader("📋 Resultados por Imagem")
                st.dataframe(resultados, use_container_width=True)

if fonte != "Dataset Local" and image_pill is not None:
    st.divider()
    col_img, col_result = st.columns([1,1], gap= "large")

    with col_img:
        st.subheader("Imagem enviada")
        st.image(image_pill, caption="Imagem enviada", use_container_width=True)
    
    with col_result:
        st.subheader("Resultado")
        with st.spinner("Analisando imagem..."):
            if arquivo_json is not None:
                try:
                    data = json.load(arquivo_json)
                    instances = data.get("instances", [])
                    total_modulos = 0
                    defeituosos = 0
                    normais = 0
                    
                    for inst in instances:
                        corners = inst.get("corners", [])
                        if not corners:
                            continue
                            
                        xs = [c["x"] for c in corners]
                        ys = [c["y"] for c in corners]
                        
                        x_min = max(0, int(min(xs)))
                        y_min = max(0, int(min(ys)))
                        x_max = min(image_pill.width, int(max(xs)))
                        y_max = min(image_pill.height, int(max(ys)))
                        
                        if x_max <= x_min or y_max <= y_min or (x_max - x_min) < 5 or (y_max - y_min) < 5:
                            continue
                            
                        crop_img = image_pill.crop((x_min, y_min, x_max, y_max))
                        classe, confianca, _ = prever(crop_img)
                        
                        total_modulos += 1
                        if classe == 'defeituoso':
                            defeituosos += 1
                        else:
                            normais += 1
                            
                    st.success('Análise das placas concluída!')
                    st.metric(label="Total de Módulos (Placas) Analisados", value=total_modulos)
                    
                    c1, c2 = st.columns(2)
                    c1.metric(label="✅ Normais", value= normais)
                    c2.metric(label="⚠️ Defeituosos", value= defeituosos)
                    
                    if defeituosos > 0:
                        st.error(f"Atenção: {defeituosos} módulo(s) defeituoso(s) detectado(s) na imagem!")
                    elif total_modulos > 0:
                        st.balloons()
                        
                except Exception as e:
                    st.error(f"Erro ao processar o arquivo JSON: {e}")
            else:
                classe, confianca, probabilidades = prever(image_pill)
            
                st.success(f'Classe: {classe}')
                st.metric(label="Confiança", value=f'{confianca:.2%}')
                
                st.divider()
                st.write("Probabilidades por classe")
                for nome, prob in zip(CLASSES, probabilidades):
                    st.progress(
                        float(prob),
                        text=f'{nome}: {prob:.2%}'
                    )

elif fonte != "Dataset Local":
    st.info("Aguardando imagem...")

# Rodapé
st.divider()
st.caption ("Desenvolvido por: Guilherme Gabriel Saldanha Pereira")