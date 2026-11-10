# Lab01 – Klasyfikacja liczby gwiazdek opinii Yelp (TF-IDF + MLP)

## Cel
Zaadaptowanie klasyfikatora z notatnika wykładowego (TF-IDF + perceptron wielowarstwowy) do przewidywania liczby gwiazdek (1–5) opinii z serwisu Yelp.

## Dane
- Zbiór: `Yelp/yelp_review_full` – 650k opinii treningowych, 5 zrównoważonych klas.
- Podział (z części treningowej, stratyfikacja, `SEED = 42`): **train 50k, val 10k, test 10k**.
- 50k zamiast limitu 100k – ograniczenie RAM Colaba (gęsta macierz cech: 50k × 10k × 4 B ≈ 2 GB).

## Pipeline
1. Ekstrakcja cech TF-IDF (`fit` tylko na train – brak wycieku danych).
2. `TensorDataset` + `DataLoader` (batch 256, tasowanie tylko dla train).
3. MLP: `Linear(10000→64) → ReLU → Linear(64→16) → ReLU → Dropout → Linear(16→5)`.
4. Trening: entropia krzyżowa, AdamW, planista lr, logowanie do W&B.

## Wyniki – podsumowanie
| Etap | Konfiguracja | val acc |
|---|---|---|
| Baseline (wykład) | unigramy, stop words usunięte, lr = 1e-4, 8 epok, cosine | 49,3% |
| Punkt 2 – najlepsze cechy | + bigramy, stop words zachowane | 52,7% |
| Punkt 3 – najlepszy model | + 20 epok, stały lr (bez planisty) | 60,7% |
| **Test (finalny)** | najlepszy model | **59,2%** |

Dodatkowo na teście: dokładność z tolerancją ±1 gwiazdka **94,4%**, średni błąd bezwzględny **0,49 gwiazdki**. Losowe zgadywanie: 20%.

## Punkt 1 – przykłady, przy których model myli się najbardziej
- Najmniejszy logit prawdziwej klasy **nie jest równoważny** najmniejszemu prawdopodobieństwu – softmax normalizuje po wszystkich logitach danej opinii. Poprawna miara: `log_softmax` prawdziwej klasy (= minus strata CE). Pokrycie obu metod: 60–85%.
- Źródła błędów: **szum w etykietach** (np. „4 big stars” z oceną 2), **opinie mieszane** (problem, potem pozytywne zakończenie), **sarkazm** – worek słów ignoruje kolejność i kontekst.
- Lepiej wytrenowany model myli się „pewniej” (P prawdziwej klasy ≈ 0,001 zamiast ≈ 0,08).

## Punkt 2 – parametry TfidfVectorizer
| Eksperyment | val acc | vs baseline |
|---|---|---|
| bigrams (+ stop words zachowane) | 52,7% | +3,4 pp |
| no_stopwords | 51,0% | +1,7 pp |
| vocab_5k | 50,2% | +0,9 pp |
| sublinear_tf | 49,7% | +0,5 pp |
| baseline | 49,3% | — |
| clean_newlines | 48,8% | −0,5 pp |
| vocab_20k | 46,8% | −2,5 pp |

- Hipoteza potwierdzona: zachowanie słów przeczących („not”, „never”) i bigramy („not good”) poprawiają wynik.
- Pozorna przewaga mniejszego słownika wynika z **niedouczenia** – większa pierwsza warstwa uczy się wolniej przy tym samym lr i liczbie epok.
- Różnice poniżej ok. 1 pp mieszczą się w niepewności pomiaru (±0,5 pp dla 10k próbek walidacyjnych).

## Punkt 3 – hiperparametry modelu i uczenia
**Etap A – proces uczenia:**
- Baseline był niedouczony – wszystkie zmiany dały +3,7 do +6,9 pp.
- Większy lr (1e-3, 3e-3): maksimum już w 1.–2. epoce, potem **przeuczenie** (luka train–val 12–24 pp).
- Mały lr + 20 epok: stabilnie, bez przeuczenia (luka 1,4 pp) → wybrany jako baza etapu B.
- SGD z momentum i dobrze dobranym lr okazał się konkurencyjny wobec AdamW (hipoteza nie potwierdzona).

**Etap B – architektura i planista lr:**
- Najlepszy: stały lr (bez planisty), 60,7% – model douczał się do ostatniej epoki.
- StepLR: −2,5 pp – zbyt wcześnie obniża lr i „zamraża” uczenie.
- Szersza sieć uczy się szybciej, ale nie osiąga wyższego wyniku i bardziej się przeucza; głębsza i płytsza – bez poprawy.
- Dropout 0,5 – bez efektu, bo w bazowym modelu prawie nie było przeuczenia.

**Etap C – finalny test:**
- Test 59,2% vs val 60,7% – lekki spadek naturalny (wybór najlepszego modelu na walidacji).
- Błędy głównie między **sąsiednimi klasami**; skrajne oceny (1 i 5, F1 ≈ 0,7) łatwiejsze niż środkowe (2–4, F1 ≈ 0,5).

## Najważniejszy wniosek
Wszystkie konfiguracje osiągają ok. **60%** – zmiany procesu uczenia i architektury wpływają głównie na szybkość i stabilność uczenia, a nie na końcowy wynik. Ograniczeniem jest **reprezentacja bag-of-words**, która ignoruje kolejność słów i kontekst. Największy zysk dała poprawa cech (bigramy), a dalszy postęp wymagałby modeli uwzględniających kontekst.