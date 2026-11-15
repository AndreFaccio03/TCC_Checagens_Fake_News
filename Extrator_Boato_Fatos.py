import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, json, requests, re
from typing import Dict, Optional
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import concurrent.futures
import json
import re
from tqdm import tqdm
import glob
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# número de requisições simultâneas (ajuste conforme GPU/CPU)
N_WORKERS = 4 

def parse_model_response(resposta_bruta: str):
    """
    Extrai e converte o JSON retornado pelo modelo,
    mesmo que venha formatado com blocos de código.
    """
    if not resposta_bruta:
        return None

    # Remove possíveis blocos de código markdown
    resposta_limpa = re.sub(r"```(?:json)?", "", resposta_bruta).strip()

    try:
        return json.loads(resposta_limpa)
    except json.JSONDecodeError as e:
        print(f"[Erro ao decodificar JSON]: {e}")
        print("Conteúdo recebido:", resposta_bruta)
        return None
def gerar_prompt(system_prompt, user_prompt, formato_saida):
    return f"""{system_prompt.strip()}
               {user_prompt.strip()}
                Responda APENAS com um JSON válido, seguindo exatamente o formato:
                {json.dumps(formato_saida, indent=2, ensure_ascii=False)}"""

class LLMConector:
    """
    Conector genérico para LLMs. Suporte atual: vLLM (API OpenAI-like)
    """
    def __init__(
        self,
        provider: str = "vllm",
        model: str = "google/gemma-3-27b-it",
        host: str = "http://localhost:8000/v1",
        temperature: float = 0
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.host = host

        if provider != "vllm":
            raise ValueError("Apenas 'vllm' é suportado nesta versão")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY não encontrada no .env")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "afinidade-docentes"
        }

        payload = {
            "model":  "meta-llama/llama-4-scout",
            "provider": {
                "order": ["together", "fireworks", "anyscale"]
            },
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = requests.post(endpoint, headers=headers, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"Erro na requisição: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]
            
def processar_comentario(index, texto):
    #print(comentario)
    user_prompt = f"""
    Com base nos textos apresentados, extraia:
    - recriação do boato que deu origem ao processo de verificação
    - lista de fatos verificados pelos jornalistas.
    A resposta deve ser apenas um JSON estruturado, conforme o formato indicado.

    texto: {texto["text_news"]}  
    """

    persona = LLMPersona(conector, system_prompt=system_prompt, exemplo_saida=formato_saida)
    resposta_bruta = persona.executar(user_prompt)

    # --- Corrige o caso em que a resposta vem duplamente empacotada ---
    if isinstance(resposta_bruta, dict) and "resposta_bruta" in resposta_bruta:
        resposta_raw = resposta_bruta["resposta_bruta"]
    else:
        resposta_raw = resposta_bruta

    try:
        resposta_extraida = extrair_json(resposta_raw)
    except Exception as e:
        resposta_extraida = {"erro_extracao": str(e), "resposta_bruta": resposta_raw}

    # --- Checa se todas as chaves esperadas estão presentes ---
    if all(k in resposta_extraida for k in formato_saida.keys()):
        resposta_final = resposta_extraida
    else:
        resposta_final = {
            "erro": "Expected key missing from classification result",
            "resposta_bruta": resposta_extraida
        }

    resposta_final["text_news"] = texto["text_news"]
    resposta_final["source_name"] = texto["source_name"]
    resposta_final["title"] = texto["title"]
    resposta_final["categories"] = texto["categories"]
    resposta_final["url"] = texto["url"]
    resposta_final["rating"] = texto["rating"]
    
    save_output(resposta_final)
    return resposta_final

def extrair_json(resposta_bruta):
    """
    Extrai JSON válido de uma string ou retorna o próprio dict se já for um.
    Remove blocos de markdown como ```json ... ``` se existirem.
    """
    if isinstance(resposta_bruta, dict):
        return resposta_bruta  # já é JSON válido

    if not isinstance(resposta_bruta, str):
        raise TypeError(f"Esperado str ou dict, recebido {type(resposta_bruta)}")

    # Remove delimitadores de markdown (```json ... ```)
    padrao = r"```(?:json)?(.*?)```"
    match = re.search(padrao, resposta_bruta, re.DOTALL)

    if match:
        resposta_limpa = match.group(1).strip()
    else:
        resposta_limpa = resposta_bruta.strip()

    try:
        return json.loads(resposta_limpa)
    except json.JSONDecodeError as e:
        raise ValueError(f"Falha ao decodificar JSON: {e}\nConteúdo:\n{resposta_limpa}")
    
class LLMPersona:
    def __init__(self, conector: LLMConector, system_prompt: str, exemplo_saida: Optional[Dict] = None):
        self.conector = conector
        self.system_prompt = system_prompt
        self.exemplo_saida = exemplo_saida

    def executar(self, user_prompt: str) -> Dict:
        if self.exemplo_saida:
            estrutura = "\n\nFormato de saída esperado (em JSON, apenas o JSON):\n" + json.dumps(self.exemplo_saida, indent=2)
        else:
            estrutura = ""

        system_prompt_final = self.system_prompt + estrutura + "\n\nA resposta deve ser apenas um JSON estruturado."
        user_prompt_final = user_prompt + estrutura + "\n\nA resposta deve ser apenas um JSON estruturado."

        resposta = self.conector.chat(system_prompt=system_prompt_final, user_prompt=user_prompt_final)

        try:
            return json.loads(resposta)
        except Exception as e:
            return {"erro": str(e), "resposta_bruta": resposta}

def save_output(classification_result: dict) -> None:
    try:
        for key in formato_saida:
            if key not in classification_result:
                raise ValueError(f"Expected key missing from classification result: '{key}'")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(classification_result, ensure_ascii=False) + "\n")

    except (ValueError, TypeError) as e:
        error_log = {
            "erro": str(e),
            "resposta_bruta": (
                classification_result
                if isinstance(classification_result, str)
                else json.dumps(classification_result, ensure_ascii=False)
            )
        }
        with open(log_file, "a", encoding="utf-7") as f:
            f.write(json.dumps(error_log, ensure_ascii=False) + "\n")


def gerar_id_unico_campos(url, title):
    return f"{url}||{title}"

    

# =========================
# Carregamento de dados
# =========================
df = pd.read_csv("central_de_fatos.csv", sep=";")
# df = df.head(10)
#print(df)

df["ID_UNICO"] = df.apply(
    lambda x: gerar_id_unico_campos(
        x["url"],
        x["title"]
    ),
    axis=1
)


print("Total de boatos:", df.shape)

system_prompt = """Você é um cientista da computação e está treinando um modelo para detecção de fake news.
                   Para ampliar seus dados de treinamento de boatos e fatos, você coletou informações de sites de checagem com o intuito de ampliar sua base de dados.
                   Esses textos possuem como característica a informação do boato que está sendo analisado, que pode ser verdadeiro ou falso, e um conjunto de fatos (refutações) que são dadas pelos jornalistas para categorizar o boato
                   Sua tarefa é extrair o conteúdo original do boato. Se o boato é exibido explicitamente no próprio texto, você deve extrai-lo na integra. 
                   Caso ele apareça indiretamente no texto, você deve coletar as informações sobre ele e recriá-lo, mantendo o máximo de informações possíveis sobre entidades, locais, assunto, etc.
                   Além disso, você deve extrair os fatos que são usados pelos jornalistas para contestação da informação.
                   A informação é composta pelo:
                   * *texto* do site de checagem, contendo o boato analisado e as refutações de jornalistas;
                   * a categoria que o boato foi enquadrado
                   Extraia o boato original e os fatos usando as seguintes regras:
                    1. NO CAMPO BOATO: use a ideia de persona inversa. Você foi quem criou o boato. O campo "boato" deve conter o boato na íntegra. Caso não possua o conteúdo original, você deve recriar o boato como se fosse uma notícia divulgada com intuito de enganar o leitor (texto no presente do indicativo, sem aspas e sem termos como "é falso que" ou "boato"). Não resuma as informações. Mantenha os nomes de instituições, valores numéricos e datas mencionadas.
                    2. NO CAMPO FATO: deve ter fidelidade aos fatos que aparecem no texto, verificados pelos jornalistas. Deve ser uma LISTA de strings contendo apenas fatos e dados numéricos coletados pelos jornalistas. Evite expressões como "o jornalista verificou que".
                    
                    Exemplo de entrada:
                    Um vídeo que circula no WhatsApp mostra pílulas de paracetamol contendo um vírus mortal de nome "Machupo". A mensagem, que já tem 50 mil compartilhamentos, 
                    diz para não tomar o remédio. Nós entramos em contato com a Anvisa e com o Ministério da Saúde. A Anvisa informou, por meio de nota enviada no dia 15 de março, 
                    que não há nenhum alerta sobre o medicamento e que o vírus Machupo, endêmico na Bolívia, não sobrevive em ambientes secos como uma pílula. Fizemos essa checagem 
                    devido ao grande volume de pedidos de leitores no nosso canal de denúncias.

                    Exemplo de saída:
                    {{
                    "boato": "Pílulas de paracetamol estão infectadas com o vírus mortal Machupo.",
                    "fatos": [
                        "Nota da Anvisa de 15 de março afirma não haver alerta sobre o medicamento paracetamol.",
                        "O vírus Machupo é endêmico na Bolívia",
                        "O vírus Machupo não sobrevive em ambientes secos, como uma pílula.",
                    ]
                    }}

                    
                    *A saída deve ser estritamente o JSON estruturado. Não inclua comentários, títulos ou qualquer outro conteúdo fora do JSON. 
                """

formato_saida = {
            "boato": "O boato original ou recriado a partir das informações contidas no texto.",
            "fatos":  ["Lista de todas as evidências factuais e dados oficiais fornecidas pelos jornalistas que checaram a informação."],
}

# Criar o conector e a persona
conector = LLMConector(model="meta-llama/llama-4-scout", temperature=0)

output_file = f"./saida-openrouter.jsonl"
log_file = "./erros_resposta.jsonl"

if os.path.exists(output_file):
    df_processado = pd.read_json(output_file, lines=True)

    # reconstruir ID_UNICO com base nos campos existentes
    df_processado["ID_UNICO"] = df_processado.apply(
        lambda x: gerar_id_unico_campos(
            x.get("url"),
            x.get("title")
        ),
        axis=1
    )

    processed = set(df_processado["ID_UNICO"].dropna())
else:
    processed = set()

# Filtrar os comentários que ainda não foram processados
comentarios_pendentes = df[~df["ID_UNICO"].isin(processed)]

with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    list(tqdm(
        executor.map(lambda x: processar_comentario(x[0], x[1]), comentarios_pendentes.iterrows()),
        total=len(comentarios_pendentes),
        desc="Processando comentários"
    ))

    