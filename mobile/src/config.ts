/**
 * URL do backend Flask (Ir Além 2 — API JSON).
 *
 * IMPORTANTE: são DOIS servidores em portas diferentes.
 *  - O Expo serve o APP (Metro) em :8081 — não mexa nisso.
 *  - O Flask serve a API em :5000 — é para cá que o app faz as requisições.
 * Portanto, aqui sempre apontamos para a porta 5000 (o backend), nunca a 8081.
 *
 * Ajuste o HOST conforme onde o app roda:
 *  - iOS Simulator / Web ............. http://localhost:5000
 *  - Emulador Android ................ http://10.0.2.2:5000
 *  - Celular físico (Expo Go) ........ http://<IP-DA-SUA-MÁQUINA>:5000
 *    (descubra com `ipconfig getifaddr en0` no macOS; o celular precisa estar
 *     na mesma rede Wi-Fi, e o Flask deve subir com host=0.0.0.0 — já é o padrão.)
 */
export const API_BASE_URL = 'http://localhost:5000';
