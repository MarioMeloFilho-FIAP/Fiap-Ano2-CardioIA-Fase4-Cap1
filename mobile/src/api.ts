import { Platform } from 'react-native';

import { API_BASE_URL } from './config';
import type { ApiResultado, Modelo, Predicao } from './types';

/**
 * Anexa a imagem ao FormData de forma correta em cada plataforma:
 *  - Web: o `uri` é um blob:/data: URL; precisamos buscar o Blob real e enviá-lo
 *    (o objeto {uri,...} viraria "[object Object]" no navegador).
 *  - Nativo (iOS/Android): o React Native aceita { uri, name, type }.
 */
async function anexarImagem(form: FormData, uri: string): Promise<void> {
  if (Platform.OS === 'web') {
    const blob = await (await fetch(uri)).blob();
    form.append('imagem', blob, 'xray.jpg');
  } else {
    form.append('imagem', { uri, name: 'xray.jpg', type: 'image/jpeg' } as unknown as Blob);
  }
}

/**
 * Type guard: valida em runtime o JSON da API antes de confiar nele.
 * O tipo do TS some na compilação; este guard protege contra resposta
 * inesperada. (Em produção, `zod` faria isso de forma declarativa.)
 */
function ehPredicao(x: unknown): x is Predicao {
  if (typeof x !== 'object' || x === null) return false;
  const o = x as Record<string, unknown>;
  return (
    typeof o.classe === 'string' &&
    typeof o.rotulo === 'string' &&
    typeof o.probabilidade === 'number' &&
    typeof o.prob_pneumonia === 'number'
  );
}

/**
 * Envia a imagem (URI local do dispositivo) ao backend e devolve a classificação.
 * Nunca lança: erros viram `{ ok: false, erro }` para a UI tratar num só lugar.
 */
export async function classificarImagem(uri: string, modelo: Modelo): Promise<ApiResultado> {
  try {
    const form = new FormData();
    await anexarImagem(form, uri);
    form.append('modelo', modelo);

    const resposta = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      body: form,
    });
    const json: unknown = await resposta.json();

    if (!resposta.ok) {
      const erro =
        typeof json === 'object' && json !== null && 'erro' in json
          ? String((json as Record<string, unknown>).erro)
          : `HTTP ${resposta.status}`;
      return { ok: false, erro };
    }

    if (!ehPredicao(json)) {
      return { ok: false, erro: 'Resposta inesperada do servidor.' };
    }
    return { ok: true, pred: json };
  } catch (e) {
    return { ok: false, erro: e instanceof Error ? e.message : 'Falha de rede.' };
  }
}
