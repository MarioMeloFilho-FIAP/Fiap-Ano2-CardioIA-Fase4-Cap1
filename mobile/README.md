# CardioIA Mobile — Ir Além 2 (React Native + Expo)

Protótipo **mobile** que consome a API do backend Flask: o usuário escolhe uma
radiografia, seleciona o modelo (Transfer Learning ou CNN do zero) e vê a
**classe**, a **confiança** e o **mapa de calor Grad-CAM**.

> Parte do projeto CardioIA — Fase 4 / Cap 1. App acadêmico de **apoio à decisão**,
> não é diagnóstico médico.

## Stack

- **Expo (SDK 52)** + **React Native 0.76** + **TypeScript** (strict).
- `expo-image-picker` para escolher a imagem da galeria.
- `fetch` + `FormData` para enviar ao endpoint `POST /api/predict`.

## Arquitetura (por que assim)

| Arquivo | Papel |
|---|---|
| `src/config.ts` | URL do backend (um único lugar para ajustar por ambiente) |
| `src/types.ts` | Contrato da API + **uniões discriminadas** (estado da tela e resultado) — o compilador obriga a tratar sucesso/erro e cada fase da UI |
| `src/api.ts` | Chamada à API; **valida o JSON em runtime** (type guard) antes de confiar; nunca lança — devolve `{ok}`/`{erro}` |
| `App.tsx` | Tela única; estado event-driven com `useState` (sem `useEffect` para derivar estado) |

## Pré-requisitos

1. **Backend rodando** (na pasta raiz do projeto):
   ```bash
   make web        # sobe o Flask em 0.0.0.0:5000 (já treine os modelos antes: make train)
   ```
2. **Node.js 18+** e o app Expo Go no celular (ou um emulador Android / iOS Simulator).

## Como rodar

```bash
cd mobile
npm install
npx expo install --fix   # alinha as versões nativas ao SDK do Expo
npx expo start           # abre o Metro; leia o QR code com o Expo Go
```

### ⚠️ Configurar a URL do backend (`src/config.ts`)

**Dois servidores, duas portas:** o Expo serve o **app** (Metro) em `:8081`, e o
Flask serve a **API** em `:5000`. O `API_BASE_URL` aponta sempre para o **backend
(5000)** — nunca para a porta do Expo (8081).

`localhost` no celular aponta para o **próprio celular**, não para o seu PC. Ajuste:

| Onde o app roda | `API_BASE_URL` |
|---|---|
| iOS Simulator / Web | `http://localhost:5000` |
| Emulador Android | `http://10.0.2.2:5000` |
| **Celular físico (Expo Go)** | `http://<IP-DA-SUA-MÁQUINA>:5000` |

Como descobrir o IP da sua máquina na rede local:

| Sistema | Comando |
|---|---|
| **macOS** | `ipconfig getifaddr en0` (Wi-Fi) ou `en1` |
| **Linux** | `hostname -I \| awk '{print $1}'` (ou `ip addr show`) |
| **Windows** | `ipconfig` → procure "Endereço IPv4" do adaptador Wi-Fi (ou, no PowerShell: `ipconfig | findstr IPv4`) |

Celular e PC precisam estar na **mesma rede Wi-Fi**. O backend já sobe com
`host=0.0.0.0`, então aceita conexões da rede local.

## Fluxo do app

1. Selecionar o modelo (chip Transfer/CNN do zero).
2. **Escolher imagem** → galeria → recorte opcional.
3. **Classificar** → chama `POST /api/predict` → exibe classe, confiança e Grad-CAM.

## Troubleshooting

**`Error: The required package 'expo-asset' cannot be found`** ao rodar `expo start`.
O `expo-asset` é um pacote core que o Metro exige. Instale com a versão do SDK e
limpe o cache:
```bash
npx expo install expo-asset
npx expo start --clear
```

**Rodar no navegador (Expo Web / Chrome Device Mode).** Os pacotes de web não vêm por
padrão:
```bash
npx expo install react-dom react-native-web
npx expo start --web
```

**`expo start` reclama de versões incompatíveis** (`react-native@... expected ...`).
Rode `npx expo install --fix` para alinhar tudo ao SDK 52.

**App abre mas falha ao classificar (erro de rede).** Quase sempre é a `API_BASE_URL`
em `src/config.ts` — veja a seção de configuração acima e confirme que o backend
(`make web`) está no ar e na mesma Wi-Fi.




