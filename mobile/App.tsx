import { useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';
import * as ImagePicker from 'expo-image-picker';

import { classificarImagem } from './src/api';
import type { EstadoTela, Modelo } from './src/types';

export default function App() {
  // Estados locais event-driven — nada de useEffect para derivar valores.
  const [imagemUri, setImagemUri] = useState<string | null>(null);
  const [modelo, setModelo] = useState<Modelo>('transfer');
  const [estado, setEstado] = useState<EstadoTela>({ fase: 'inicial' });

  const escolherImagem = async () => {
    const resultado = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      quality: 1,
    });
    // noUncheckedIndexedAccess: assets[0] pode ser undefined — checamos antes.
    const asset = resultado.canceled ? undefined : resultado.assets[0];
    if (asset) {
      setImagemUri(asset.uri);
      setEstado({ fase: 'inicial' }); // limpa resultado anterior
    }
  };

  const analisar = async () => {
    if (!imagemUri) return;
    setEstado({ fase: 'carregando' });
    const res = await classificarImagem(imagemUri, modelo);
    setEstado(res.ok ? { fase: 'ok', pred: res.pred } : { fase: 'erro', mensagem: res.erro });
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <StatusBar style="light" />

      <View style={styles.header}>
        <Text style={styles.titulo}>CardioIA</Text>
        <Text style={styles.subtitulo}>Triagem visual de raios-X</Text>
      </View>

      {/* Seletor de modelo */}
      <View style={styles.modelos}>
        {(['transfer', 'scratch'] as const).map((m) => (
          <Pressable
            key={m}
            onPress={() => setModelo(m)}
            style={[styles.chip, modelo === m && styles.chipAtivo]}
          >
            <Text style={[styles.chipTexto, modelo === m && styles.chipTextoAtivo]}>
              {m === 'transfer' ? 'Transfer (VGG16)' : 'CNN do zero'}
            </Text>
          </Pressable>
        ))}
      </View>

      <Pressable style={styles.botao} onPress={escolherImagem}>
        <Text style={styles.botaoTexto}>Escolher imagem</Text>
      </Pressable>

      {imagemUri && (
        <Image source={{ uri: imagemUri }} style={styles.preview} resizeMode="contain" />
      )}

      <Pressable
        style={[styles.botao, (!imagemUri || estado.fase === 'carregando') && styles.botaoDesabilitado]}
        onPress={analisar}
        disabled={!imagemUri || estado.fase === 'carregando'}
      >
        <Text style={styles.botaoTexto}>Classificar</Text>
      </Pressable>

      {/* Render por fase — a união discriminada garante que cobrimos todos os casos */}
      {estado.fase === 'carregando' && (
        <ActivityIndicator size="large" color="#1565c0" style={{ marginTop: 24 }} />
      )}

      {estado.fase === 'erro' && (
        <View style={styles.erroBox}>
          <Text style={styles.erroTexto}>{estado.mensagem}</Text>
        </View>
      )}

      {estado.fase === 'ok' && (
        <View
          style={[
            styles.resultado,
            estado.pred.classe === 'PNEUMONIA' ? styles.resPneumonia : styles.resNormal,
          ]}
        >
          <Text style={styles.resClasse}>{estado.pred.classe}</Text>
          <Text style={styles.resRotulo}>{estado.pred.rotulo}</Text>
          <Text style={styles.resConf}>
            Confiança: {(estado.pred.probabilidade * 100).toFixed(1)}%
          </Text>
          {estado.pred.gradcam && (
            <>
              <Text style={styles.gradcamLegenda}>Grad-CAM (onde o modelo olhou)</Text>
              <Image source={{ uri: estado.pred.gradcam }} style={styles.gradcam} />
            </>
          )}
        </View>
      )}

      <Text style={styles.aviso}>
        ⚠️ Protótipo acadêmico de apoio à decisão. Não é diagnóstico médico.
      </Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, paddingTop: 60, backgroundColor: '#f4f6f9', flexGrow: 1 },
  header: { alignItems: 'center', marginBottom: 20 },
  titulo: { fontSize: 30, fontWeight: '800', color: '#0d3c75' },
  subtitulo: { fontSize: 15, color: '#5b6470' },
  modelos: { flexDirection: 'row', justifyContent: 'center', gap: 10, marginBottom: 16 },
  chip: {
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#1565c0',
  },
  chipAtivo: { backgroundColor: '#1565c0' },
  chipTexto: { color: '#1565c0', fontWeight: '600' },
  chipTextoAtivo: { color: '#fff' },
  botao: {
    backgroundColor: '#1565c0',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
    marginTop: 12,
  },
  botaoDesabilitado: { backgroundColor: '#9bb4d1' },
  botaoTexto: { color: '#fff', fontSize: 16, fontWeight: '700' },
  preview: { width: '100%', height: 260, borderRadius: 10, marginTop: 16 },
  erroBox: { backgroundColor: '#fdecea', borderRadius: 10, padding: 14, marginTop: 20 },
  erroTexto: { color: '#c62828' },
  resultado: { borderRadius: 12, padding: 20, marginTop: 20, alignItems: 'center' },
  resNormal: { backgroundColor: '#2e7d32' },
  resPneumonia: { backgroundColor: '#c62828' },
  resClasse: { color: '#fff', fontSize: 24, fontWeight: '800', letterSpacing: 0.5 },
  resRotulo: { color: '#fff', fontSize: 15, opacity: 0.95, marginTop: 2 },
  resConf: { color: '#fff', fontSize: 15, marginTop: 6 },
  gradcamLegenda: { color: '#fff', fontSize: 13, marginTop: 16, opacity: 0.9 },
  gradcam: { width: 220, height: 220, borderRadius: 8, marginTop: 8 },
  aviso: { color: '#5b6470', fontSize: 12, textAlign: 'center', marginTop: 28 },
});
