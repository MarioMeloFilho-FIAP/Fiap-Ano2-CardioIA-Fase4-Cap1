/** Contrato de dados entre o app mobile e a API Flask. */

// União de literais em vez de enum (mais simples e tree-shakeable).
export type Modelo = 'transfer' | 'scratch';

/** Resposta de /api/predict (espelha o payload do backend). */
export interface Predicao {
  classe: string; // "NORMAL" | "PNEUMONIA"
  rotulo: string;
  probabilidade: number; // confiança da classe vencedora, [0,1]
  prob_pneumonia: number; // P(PNEUMONIA), [0,1]
  modelo: string;
  classes: string[];
  gradcam?: string; // data URI base64 do mapa de calor
}

/**
 * Resultado da chamada à API como UNIÃO DISCRIMINADA: o compilador obriga a
 * tratar sucesso e erro — nada de checar `if (res.erro)` e esquecer um caminho.
 */
export type ApiResultado =
  | { ok: true; pred: Predicao }
  | { ok: false; erro: string };

/**
 * Estado da tela, também como união discriminada. Cada fase carrega só os dados
 * que fazem sentido nela (ex.: 'erro' tem mensagem; 'ok' tem a predição).
 */
export type EstadoTela =
  | { fase: 'inicial' }
  | { fase: 'carregando' }
  | { fase: 'ok'; pred: Predicao }
  | { fase: 'erro'; mensagem: string };
