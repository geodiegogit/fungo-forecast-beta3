import os
import json
import math
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

ENDPOINT_SENSORS_ANAGRAFICA = "https://www.dati.lombardia.it/resource/nf78-nj6b.json"
ENDPOINT_METEO_DATA = "https://www.dati.lombardia.it/resource/647i-nhxk.json"
ID_STAZIONE = "1545"
QUOTA_STAZIONE = 1285

FINESTRA_SENESCENZA_GG = 15          # invece di guardare tutti i 90 gg di storico
GIORNI_STORICO_INDICI = 180          # quanti giorni di indici teniamo per la taratura


# ---------------------------------------------------------------------------
# Utility di rete: le chiamate ad ARPA a volte falliscono/timeoutano; prima
# non c'era nessun retry, quindi un singolo intoppo di rete faceva fallire
# tutto l'aggiornamento notturno.
# ---------------------------------------------------------------------------
def http_get_json(url: str, params: Dict[str, Any], tentativi: int = 3, timeout: int = 25) -> Any:
    ultimo_errore: Optional[Exception] = None
    for tentativo in range(1, tentativi + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            ultimo_errore = e
            print(f"[WARN] Tentativo {tentativo}/{tentativi} fallito per {url}: {e}")
            if tentativo < tentativi:
                time.sleep(2 ** tentativo)
    raise RuntimeError(f"Impossibile contattare {url} dopo {tentativi} tentativi: {ultimo_errore}")


def get_sensor_ids_for_station(id_stazione: str) -> Dict[str, str]:
    """Recupera un sensore per tipo (pioggia/temperatura/umidita/vento).

    Se per la stazione esistono più sensori dello stesso tipo (es. uno storico
    dismesso e uno attivo), preferiamo quello con storico == 'A'. Se il campo
    'storico' non è presente nel dataset, il comportamento resta quello di
    prima (ultimo trovato) ma logghiamo il caso per verifica manuale.
    """
    params = {"$where": f"idstazione = '{id_stazione}'", "$limit": 50}
    dati = http_get_json(ENDPOINT_SENSORS_ANAGRAFICA, params)

    mappa: Dict[str, str] = {}
    stato_scelto: Dict[str, str] = {}
    duplicati = []

    for s in dati:
        tipo = s.get("tipologia", "").lower()
        stato = str(s.get("storico", "")).upper()  # 'A' = attivo, se il campo esiste
        ids = s.get("idsensore")

        categoria = None
        if "precipitazione" in tipo:
            categoria = "pioggia"
        elif "temperatura" in tipo:
            categoria = "temperatura"
        elif "umidit" in tipo:
            categoria = "umidita"
        elif "direzione" in tipo:
            # La "Direzione Vento" (gradi 0-360) contiene spesso la parola
            # "vento" nella tipologia esattamente come "Velocità Vento":
            # va esclusa esplicitamente, altrimenti rischia di sovrascrivere
            # il sensore di velocità (bug osservato: vento_max > 1000 km/h,
            # in realtà erano gradi moltiplicati per 3.6).
            continue
        elif "vento" in tipo or "velocit" in tipo:
            categoria = "vento"
        if categoria is None or ids is None:
            continue

        if categoria not in mappa:
            mappa[categoria] = ids
            stato_scelto[categoria] = stato
        else:
            duplicati.append(categoria)
            # Se il nuovo sensore è esplicitamente attivo e quello già scelto no,
            # lo sostituiamo. Se non abbiamo informazioni sullo stato, teniamo
            # il primo trovato (comportamento precedente).
            if stato == "A" and stato_scelto.get(categoria) != "A":
                mappa[categoria] = ids
                stato_scelto[categoria] = stato

    if duplicati:
        print(f"[ATTENZIONE] Sensori multipli trovati per: {sorted(set(duplicati))}. "
              f"Verifica manualmente idsensore scelto: {mappa}")

    return mappa


def download_weather_history(mappa_sensori: Dict[str, str], days: int = 45) -> pd.DataFrame:
    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
    id_list = "','".join(mappa_sensori.values())
    params = {
        "$where": f"idsensore in ('{id_list}') AND data >= '{start_date}' AND valore != -9999",
        "$limit": 50000, "$order": "data ASC"
    }
    records = http_get_json(ENDPOINT_METEO_DATA, params)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["data"] = pd.to_datetime(df["data"])
    df["valore"] = pd.to_numeric(df["valore"], errors="coerce")
    inv_map = {v: k for k, v in mappa_sensori.items()}
    df["tipo"] = df["idsensore"].map(inv_map)
    return df.pivot_table(index="data", columns="tipo", values="valore", aggfunc="mean").reset_index()


def aggregate_daily(df_hourly: pd.DataFrame) -> List[Dict[str, Any]]:
    if df_hourly.empty:
        return []

    df_hourly["giorno"] = df_hourly["data"].dt.strftime("%Y-%m-%d")
    oggi_str = datetime.now().strftime("%Y-%m-%d")
    df_hourly = df_hourly[df_hourly["giorno"] < oggi_str]

    if df_hourly.empty:
        return []

    agg_rules = {}
    if "pioggia" in df_hourly.columns: agg_rules["pioggia"] = "sum"
    if "umidita" in df_hourly.columns: agg_rules["umidita"] = "mean"
    if "vento" in df_hourly.columns: agg_rules["vento"] = "max"
    if "temperatura" in df_hourly.columns: agg_rules["temperatura"] = ["mean", "max", "min"]

    df_daily = df_hourly.groupby("giorno").agg(agg_rules)
    df_daily.columns = ['_'.join(col).strip() for col in df_daily.columns.values]
    df_daily = df_daily.reset_index()

    serie = []
    for _, row in df_daily.iterrows():
        vento_kmh = row.get("vento_max", 0.0) * 3.6 if pd.notna(row.get("vento_max")) else 10.0
        if vento_kmh > 200.0:
            # Un vento sopra i 200 km/h è fisicamente assurdo per la zona:
            # quasi certamente il sensore mappato non è la velocità del vento
            # (es. sta leggendo la direzione in gradi). Segnaliamolo forte nei
            # log invece di pubblicare dati palesemente sbagliati in silenzio.
            print(f"[ATTENZIONE] vento_max sospetto ({vento_kmh:.1f} km/h) il {row['giorno']}: "
                  f"controllare il mapping del sensore 'vento'.")
        serie.append({
            "data": row["giorno"],
            "pioggia_mm": round(float(row.get("pioggia_sum", 0.0)), 1),
            "t_media": round(float(row.get("temperatura_mean", 15.0)), 1),
            "t_max": round(float(row.get("temperatura_max", 15.0)), 1),
            "t_min": round(float(row.get("temperatura_min", 15.0)), 1),
            "rh_media": round(float(row.get("umidita_mean", 70.0)), 1),
            "vento_max": round(float(vento_kmh), 1)
        })
    return serie


def stima_et_giornaliera(t_media: float) -> float:
    """Stima molto semplificata dell'evapotraspirazione potenziale (mm/giorno).

    Non è un vero Thornthwaite/Penman (richiederebbero dati di radiazione e
    durata del giorno che non abbiamo), ma una curva monotona crescente con
    la temperatura pensata per dare un'idea approssimativa di quanta della
    pioggia caduta viene "restituita" all'atmosfera invece di restare nel
    terreno. Usata solo a scopo informativo/diagnostico: la classificazione
    della siccità usa invece i "giorni caldo-secco" (vedi sotto), che si sono
    rivelati più robusti su un caso reale di anticiclone prolungato dove il
    bilancio pioggia-ET risultava quasi in pareggio pur in presenza di stress
    vegetativo evidente (senescenza fogliare anticipata).
    """
    if t_media <= 0:
        return 0.0
    return 0.02 * (t_media ** 1.6)


def conta_giorni_caldo_secco(serie: List[Dict[str, Any]], ini: int, fine: int,
                              soglia_pioggia: float = 2.0, soglia_tmax: float = 24.0) -> int:
    """Conta, nella finestra [ini, fine), i giorni con pochissima pioggia e
    caldo: è il proxy usato per riconoscere un anticiclone prolungato (tipo
    quello africano) che stressa i boschi anche quando il bilancio pioggia-ET
    non risulta drammaticamente negativo (perché qualche pioggia sporadica
    'spezza' il conteggio ma non basta a reidratare davvero il terreno)."""
    ini = max(0, ini)
    fine = max(ini, fine)
    return sum(1 for k in range(ini, fine)
               if serie[k]["pioggia_mm"] < soglia_pioggia and serie[k].get("t_max", 0) >= soglia_tmax)


class AnalizzatoreSiccitaPorcini:
    GIORNI_FINESTRA_SICCITA = 35   # ampiezza della finestra di "memoria" del bosco

    def __init__(self, soglia_evento: float = 35.0):
        self.soglia_evento = soglia_evento

    def _soglia_e_smorzamento(self, giorni_caldo_secco: int):
        """Più lungo è stato lo stress caldo-secco pregresso, più grande deve
        essere la pioggia per considerarsi un vero 'reset' della siccità, e
        più la buttata risultante resta comunque ritardata e attenuata anche
        quando l'evento la supera (il micelio/l'albero non si riprendono
        istantaneamente). Caso reale che ha guidato la taratura: dopo ~5
        settimane di anticiclone (17-18 giorni caldo-secco su 35), una pioggia
        di 55.8mm concentrata in un solo giorno non ha prodotto nulla, mentre
        un evento di 93mm distribuito su 3 giorni pochi giorni dopo sì."""
        if giorni_caldo_secco >= 12:
            return 70.0, 10, 0.50    # siccità severa e prolungata (es. anticiclone africano)
        if giorni_caldo_secco >= 6:
            return 45.0, 5, 0.75     # siccità moderata
        return self.soglia_evento, 0, 1.00   # condizioni normali

    def analizza(self, serie: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not serie:
            return {"evento_rilevato": False, "stato": "In attesa di dati ARPA"}

        n = len(serie)
        giorno_ieri = serie[-1]

        eventi_trovati = []
        i = 2
        while i < n:
            finestra = serie[i - 2:i + 1]
            c3 = sum(g["pioggia_mm"] for g in finestra)

            # Individuiamo il giorno di picco DIRETTAMENTE dall'indice nella
            # finestra (invece di ri-cercarlo con list.index sui valori, che
            # in teoria può confondersi con giorni dai valori identici).
            idx_rel, giorno_max = max(enumerate(finestra), key=lambda t: t[1]["pioggia_mm"])
            idx = i - 2 + idx_rel

            giorni_cs = conta_giorni_caldo_secco(serie, idx - self.GIORNI_FINESTRA_SICCITA - 2, idx - 2)
            soglia_dinamica, ritardo, smorz = self._soglia_e_smorzamento(giorni_cs)

            if c3 >= soglia_dinamica:
                if not eventi_trovati or (idx - eventi_trovati[-1]["indice"]) >= 4:
                    giorni_da_ev = (n - 1) - idx

                    giorni_favonio = sum(1 for k in range(idx + 1, n)
                                          if serie[k]["vento_max"] > 22.0
                                          and serie[k]["rh_media"] < 60.0
                                          and serie[k]["pioggia_mm"] < 1.0)
                    danno_favonio = max(0.1, 1.0 - (giorni_favonio * 0.25))

                    eventi_trovati.append({
                        "indice": idx,
                        "data": giorno_max["data"],
                        "pioggia": c3,
                        "giorni_caldo_secco_pregressi": giorni_cs,
                        "giorni_da_evento": giorni_da_ev,
                        "ritardo": ritardo,
                        "soglia": soglia_dinamica,
                        "smorzamento": smorz * danno_favonio
                    })
                i += 3
            else:
                i += 1

        eventi_trovati = [ev for ev in eventi_trovati if ev["giorni_da_evento"] <= 40]

        # Notti tropicali SOLO nella finestra recente (prima guardava tutti i
        # 90 gg di storico, quindi un caldo di agosto restava "attivo" come
        # allerta anche a ottobre).
        finestra_recente = serie[-FINESTRA_SENESCENZA_GG:]
        notti_tropicali = sum(1 for d in finestra_recente if d.get("t_min", 0) >= 19.0)

        # Siccità prolungata "ad oggi": stessa logica usata per gli eventi,
        # ma calcolata sugli ultimi giorni della serie invece che prima di un
        # evento di pioggia. È quella che ha causato la senescenza fogliare
        # anticipata delle betulle quest'anno (anticiclone africano).
        giorni_caldo_secco_attuali = conta_giorni_caldo_secco(
            serie, n - self.GIORNI_FINESTRA_SICCITA, n)
        siccita_prolungata = giorni_caldo_secco_attuali >= 12

        rischio_senescenza = notti_tropicali >= 2 or siccita_prolungata

        diag = {
            "t_max_attuale": giorno_ieri["t_max"],
            "t_min_attuale": giorno_ieri["t_min"],
            "rh_media_attuale": giorno_ieri["rh_media"],
            "vento_max_attuale": giorno_ieri["vento_max"],
            "pioggia_oggi": giorno_ieri["pioggia_mm"],
            "eventi": eventi_trovati,
            "rischio_senescenza": rischio_senescenza,
            "notti_tropicali_15gg": notti_tropicali,
            "giorni_caldo_secco_35gg": giorni_caldo_secco_attuali,
            "siccita_prolungata": siccita_prolungata
        }

        if not eventi_trovati:
            diag.update({"evento_rilevato": False, "stato": "Nessuna pioggia rilevante"})
        else:
            ev_recente = eventi_trovati[-1]
            diag.update({
                "evento_rilevato": True,
                "data_evento": ev_recente["data"],
                "giorni_da_evento": ev_recente["giorni_da_evento"],
                "ritardo_siccita_applicato": ev_recente["ritardo"]
            })
        return diag


def calcola_microzone(diag: Dict[str, Any], quota_stazione: int = QUOTA_STAZIONE) -> List[Dict[str, Any]]:
    zone_cfg = [
        {"nome": "Camnasco", "quota": 750, "essenza": "castagno", "esposizione": "SE", "giorni_base": 6},
        {"nome": "Betulle SE", "quota": 1222, "essenza": "betulla", "esposizione": "SE", "giorni_base": 9},
        {"nome": "Betulle NE", "quota": 1144, "essenza": "betulla", "esposizione": "NE", "giorni_base": 8},
        {"nome": "Faggi Ovest", "quota": 1561, "essenza": "faggio", "esposizione": "OVEST_OMBRA", "giorni_base": 12},
        {"nome": "Abeti Nord", "quota": 1478, "essenza": "pino", "esposizione": "NORD", "giorni_base": 13}
    ]

    if not diag.get("evento_rilevato"):
        return [{"zona": z["nome"], "indice_buttata": 0.0, "stato": "In attesa",
                  "giorni_mancanti_al_picco": None, "onde": []} for z in zone_cfg]

    # Condizioni "da inversione notturna": cielo sereno, vento calmo, niente
    # pioggia. In queste notti l'aria fredda ristagna nei fondovalle, quindi
    # le quote basse possono essere PIÙ fredde del previsto rispetto al
    # semplice gradiente adiabatico standard usato sotto.
    condizioni_serene = (diag["vento_max_attuale"] < 10.0
                          and diag["rh_media_attuale"] < 75.0
                          and diag["pioggia_oggi"] < 1.0)

    res = []
    for z in zone_cfg:
        gradiente = 0.0065 * (z["quota"] - quota_stazione)
        t_min_b = diag["t_min_attuale"] - gradiente
        t_max_b = diag["t_max_attuale"] - gradiente

        rh_eff = diag["rh_media_attuale"]
        if z["esposizione"] == "SE":
            t_max_eff, t_min_eff, rh_eff = t_max_b + 2.5, t_min_b, max(0.0, diag["rh_media_attuale"] - 10.0)
        elif z["esposizione"] == "NE":
            t_max_eff, t_min_eff, rh_eff = t_max_b - 1.0, t_min_b, min(100.0, diag["rh_media_attuale"] + 12.0)
        elif z["esposizione"] == "OVEST_OMBRA":
            t_max_eff, t_min_eff, rh_eff = t_max_b - 1.5, t_min_b - 0.5, min(100.0, diag["rh_media_attuale"] + 15.0)
        elif z["esposizione"] == "NORD":
            t_max_eff, t_min_eff, rh_eff = t_max_b - 2.5, t_min_b - 1.5, min(100.0, diag["rh_media_attuale"] + 20.0)
        else:
            t_max_eff, t_min_eff = t_max_b, t_min_b

        # Correzioni speciali ecosistemi
        if z["nome"] == "Camnasco": rh_eff = max(60.0, rh_eff)
        if z["nome"] == "Faggi Ovest": t_min_eff += 1.0

        # Correzione da inversione notturna: solo per zone più basse della
        # stazione e solo in notti serene/calme.
        correzione_inversione = 0.0
        if condizioni_serene and z["quota"] < quota_stazione:
            dislivello = quota_stazione - z["quota"]
            correzione_inversione = min(3.0, 0.002 * dislivello)
            t_min_eff -= correzione_inversione

        t_opt = 16.5 if z["essenza"] not in ["pino", "faggio"] else 14.5
        t_media_eff = (t_max_eff + t_min_eff) / 2
        f_T_media = math.exp(- ((t_media_eff - t_opt) ** 2) / (2 * (3.5 ** 2)))
        f_T_freddo = 0.0 if t_min_eff < 3.0 else ((t_min_eff - 3.0) / 4.0 if t_min_eff < 7.0 else 1.0)

        # Grilletto termico
        if 8.0 <= t_min_eff <= 13.0: f_grilletto = 1.3
        elif t_min_eff > 17.0: f_grilletto = 0.7
        else: f_grilletto = 1.0

        f_H = 1.0 if rh_eff >= 85 else (0.0 if rh_eff < 40 else ((rh_eff - 40) / 45) ** 1.2)

        vento = diag["vento_max_attuale"]
        is_favonio = (vento > 20 and diag["rh_media_attuale"] < 60 and diag["pioggia_oggi"] < 1.0)

        # Sant'Amate è protetta dal catino roccioso
        if z["nome"] == "Faggi Ovest": phi_vento = 1.0
        else: phi_vento = max(0.1, 1.0 - 0.04 * (vento - 20)) if is_favonio else 1.0

        indice_totale = 0.0
        onde = []

        for ev in diag["eventi"]:
            f_R = 1.0 / (1.0 + math.exp(-0.12 * (ev["pioggia"] - ev["soglia"])))

            ritardo_applicato = ev["ritardo"]
            if z["nome"] == "Camnasco": ritardo_applicato = min(1, ritardo_applicato)

            picco_eff = z["giorni_base"] + ritardo_applicato
            f_L = math.exp(- ((ev["giorni_da_evento"] - picco_eff) ** 2) / (2 * (2.2 ** 2)))

            ind_pieno = 100.0 * (f_R * (f_T_media * f_T_freddo) * 1.0 * f_H) * phi_vento * ev["smorzamento"] * f_grilletto
            ind_oggi = ind_pieno * f_L

            indice_totale += ind_oggi
            onde.append({
                "giorni_mancanti_da_ieri": round(picco_eff - ev["giorni_da_evento"], 1),
                "indice_picco": round(ind_pieno, 1)
            })

        indice_totale = min(100.0, indice_totale)

        futuri = [onda["giorni_mancanti_da_ieri"] for onda in onde if onda["giorni_mancanti_da_ieri"] > 0]
        giorni_mancanti = min(futuri) if futuri else (min([o["giorni_mancanti_da_ieri"] for o in onde]) if onde else 0)

        if f_T_freddo == 0.0: stato = "Blocco freddo"
        elif indice_totale > 65: stato = "Buttata in corso (Onde multiple)" if len(onde) > 1 else "Buttata in corso"
        elif giorni_mancanti > 0: stato = f"Incubazione ({giorni_mancanti} gg)"
        else: stato = "In esaurimento"

        res.append({
            "zona": z["nome"],
            "quota_m": z["quota"],
            "esposizione": z["esposizione"],
            "essenza": z["essenza"],
            "indice_buttata": round(indice_totale, 1),
            "t_min_stimata": round(t_min_eff, 1),
            "t_max_stimata": round(t_max_eff, 1),
            "correzione_inversione": round(correzione_inversione, 1),
            "giorni_mancanti_al_picco": giorni_mancanti,
            "stato": stato,
            "onde": onde
        })
    return res


def aggiorna_storico_indici(data_riferimento: str, zone: List[Dict[str, Any]],
                             path: str = "data/storico_indici.json",
                             giorni_max: int = GIORNI_STORICO_INDICI) -> List[Dict[str, Any]]:
    """Tiene uno storico giornaliero dell'indice_buttata per ogni zona.

    Serve a poter confrontare in futuro, per ciascun giorno, "cosa prevedeva
    il modello" con "cosa hai davvero trovato" (i ritrovamenti salvati sul
    telefono) — ed è anche il file più utile per un confronto A/B fra questa
    app e la nuova che stai per creare.
    """
    storico: List[Dict[str, Any]] = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                storico = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        storico = []

    riga = {"data": data_riferimento, "zone": {z["zona"]: z["indice_buttata"] for z in zone}}
    storico = [r for r in storico if r.get("data") != data_riferimento]
    storico.append(riga)
    storico.sort(key=lambda r: r["data"])
    storico = storico[-giorni_max:]

    with open(path, "w", encoding="utf-8") as f:
        json.dump(storico, f, ensure_ascii=False, indent=2)
    return storico


def gestisci_stato_stagionale(data_riferimento: str, siccita_prolungata_oggi: bool,
                               giorni_caldo_secco_oggi: int,
                               path: str = "data/stagione.json") -> Dict[str, Any]:
    """Il Boletus edulis è un simbionte micorrizico: dipende dagli zuccheri
    ceduti dalla pianta ospite via le radici, non solo dall'umidità del
    terreno. Se l'albero va in stress idrico e perde anticipatamente parte
    delle foglie (osservato negli ultimi due anni, mai prima), la sua
    capacità fotosintetica resta ridotta per TUTTO il resto della stagione:
    le foglie perse non ricrescono ad agosto/settembre, anche se poi piove
    tanto e il terreno si reidrata.

    Per questo la "senescenza confermata" va ricordata in un file persistente
    e non si disattiva più con una pioggia successiva — a differenza della
    soglia dinamica sugli eventi di pioggia, che invece è correttamente
    reversibile (quella riguarda l'acqua nel suolo, non la fisiologia della
    pianta). Lo stato si azzera solo al cambio di anno solare.
    """
    anno = data_riferimento[:4]
    stato = {"anno": anno, "senescenza_confermata": False,
             "data_rilevamento": None, "giorni_caldo_secco_max": 0}

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                salvato = json.load(f)
            if salvato.get("anno") == anno:
                stato = salvato
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    stato["giorni_caldo_secco_max"] = max(stato.get("giorni_caldo_secco_max", 0), giorni_caldo_secco_oggi)

    if siccita_prolungata_oggi and not stato["senescenza_confermata"]:
        stato["senescenza_confermata"] = True
        stato["data_rilevamento"] = data_riferimento
        print(f"[INFO] Senescenza fogliare confermata per la stagione {anno} "
              f"il {data_riferimento} (giorni caldo-secco: {giorni_caldo_secco_oggi}).")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(stato, f, ensure_ascii=False, indent=2)
    return stato


def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connessione ARPA...")

    try:
        sensori = get_sensor_ids_for_station(ID_STAZIONE)
        df_orari = download_weather_history(sensori, days=45)
    except RuntimeError as e:
        print(f"[ERRORE] {e}")
        print("Aggiornamento saltato per oggi: i dati pubblicati restano quelli del giorno precedente.")
        return

    nuovi_dati = aggregate_daily(df_orari)

    if not nuovi_dati:
        print("Errore: nessun dato scaricato da ARPA (o tutti i giorni erano incompleti).")
        return

    # Carichiamo il vecchio storico (se esiste) per allungare la memoria dell'app
    storico_esistente = []
    try:
        if os.path.exists("data/storico.json"):
            with open("data/storico.json", "r") as f:
                storico_esistente = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fusione dei dati: sovrascrive i giorni duplicati aggiornandoli e tiene i vecchi
    storico_unito = {d["data"]: d for d in storico_esistente if "data" in d}
    for d in nuovi_dati:
        storico_unito[d["data"]] = d

    # Ordiniamo cronologicamente e teniamo gli ultimi 90 giorni
    storico_ordinato = sorted(storico_unito.values(), key=lambda x: x["data"])
    storico_finale = storico_ordinato[-90:]

    analizzatore = AnalizzatoreSiccitaPorcini()
    diagnosi = analizzatore.analizza(storico_finale)

    os.makedirs("data", exist_ok=True)
    stato_stagionale = gestisci_stato_stagionale(
        storico_finale[-1]["data"],
        diagnosi.get("siccita_prolungata", False),
        diagnosi.get("giorni_caldo_secco_35gg", 0)
    )
    diagnosi["senescenza_confermata_stagione"] = stato_stagionale["senescenza_confermata"]
    diagnosi["senescenza_data_rilevamento"] = stato_stagionale["data_rilevamento"]
    # Una volta confermata, la senescenza resta "vera" per tutta la stagione
    # anche se il conteggio mobile dei giorni caldo-secco scende sotto
    # soglia dopo piogge successive: la pianta non recupera le foglie perse.
    if stato_stagionale["senescenza_confermata"]:
        diagnosi["rischio_senescenza"] = True

    previsioni = calcola_microzone(diagnosi)

    storico_indici = aggiorna_storico_indici(storico_finale[-1]["data"], previsioni)

    output = {
        "ultimo_aggiornamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stazione": {"id": ID_STAZIONE, "nome": "San Siro", "quota_m": QUOTA_STAZIONE},
        "diagnosi_meteo": diagnosi,
        "zone": previsioni,
        "storico_completo": storico_finale,
        "storico_indici": storico_indici
    }

    with open("data/previsioni.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open("data/storico.json", "w", encoding="utf-8") as f:
        json.dump(storico_finale, f, ensure_ascii=False, indent=2)

    print(f"[OK] Aggiornamento completato per il {storico_finale[-1]['data']}.")


if __name__ == "__main__":
    main()
