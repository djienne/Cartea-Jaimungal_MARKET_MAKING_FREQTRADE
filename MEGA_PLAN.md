J’ai revu le bundle et la doc officielle Freqtrade stable. Mon verdict est assez clair : **le projet est beaucoup plus sûr qu’une version naïve**, avec de vrais garde-fous, mais **il ne doit pas encore trader en live via Freqtrade**. Les artefacts du repo disent eux-mêmes que les gates locales passent, mais que le déploiement reste bloqué par quatre gates externes : post-only/Alo Hyperliquid, fee-tier réel, replay multi-jours, et live canary.  Le document de statut conclut aussi que le projet reste une implémentation research/dry-run fail-closed tant que ces preuves n’existent pas. 

## Points solides déjà présents

Le code a une bonne philosophie fail-closed : `trading_enabled=false` par défaut, dry-run activé dans la config, stake faible, fee alignée, force-entry désactivé, et des gates de sécurité automatisées. Le `config_safety_report` montre `dry_run: true`, `stake_amount: 25`, `tradable_balance_ratio: 0.1`, `fee: 0.00015`, `custom_price_max_distance_ratio: 0.05`, et `order_time_in_force` en `GTC` pour le mode research/dry-run. 

Les callbacks critiques existent et testent beaucoup de cas importants : rejet si trading désactivé, rejet des shorts en mode long-only, rejet des ordres market pour les quotes passives, vérification du tick price, vérification du lot size, rejet des stale params, rejet des fee mismatches, rejet des TIF non post-only quand `post_only_verified=True`, kill switch sur position short inattendue, et logging de quote/order/fill.   

La logique `_quote_state_valid()` est bien structurée : elle refuse si trading désactivé, pas de HJB cache, HJB stale, params invalides, fee mismatch, orderbook stale, collector stale, short inattendu, côté HJB désactivé à la frontière, limite d’inventaire atteinte, live non post-only, ou gates live manquantes. C’est exactement le bon endroit pour concentrer la sécurité finale. 

## Blockers avant live

| Sévérité | Sujet                       | Problème                                                                                                                                                                                                                           | Pourquoi c’est dangereux                                                                                        | Action                                                                                                                                      |
| -------: | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Critique | Post-only / maker safety    | Hyperliquid `Alo` n’est pas prouvé via Freqtrade, et le rapport post-only est `ok=false` avec `missing_crossing_result` et `missing_passive_result`.                                                                               | Sans post-only natif, une quote GTC peut devenir taker si le marché bouge ou si l’arrondi traverse le book.     | Garder Freqtrade en dry-run/research. Live seulement via preuve `Alo`, ou via executor direct Hyperliquid SDK.                              |
| Critique | Evidence fee tier           | Le rapport fee est `ok=false` : exchange fee non prouvée, 0 maker fill, 0 actual maker fee match.                                                                                                                                  | Le modèle peut être rentable en théorie mais négatif net fees si le tier réel diffère.                          | Capturer `userFees` + fills maker réels avec fee payé et fee rate.                                                                          |
| Critique | Replay                      | Le replay acceptance échoue : pas assez de couverture, pas de maker fills, stale quote cancel ratio trop élevé.                                                                                                                    | Le dry-run Freqtrade n’est pas une simulation de queue maker.                                                   | Exiger plusieurs jours de données et calibration fills/markout.                                                                             |
| Critique | Live canary                 | `live_canary_report` est bloqué par les dépendances et indique notamment no live health, unknown liquidity fills, accepted quotes non live/non post-only.                                                                          | Aucune preuve bout-en-bout que les fills live sont maker-only et rattachés aux quotes.                          | Canary seulement après post-only, fee, replay.                                                                                              |
|     High | Signature Freqtrade         | `custom_stake_amount()` manque l’argument `leverage`. La doc stable Freqtrade actuelle inclut `leverage` entre `max_stake` et `entry_tag`. ([Freqtrade][1]) Le code/test actuel appelle la méthode sans `leverage`.                | Selon la version Freqtrade exacte, cela peut casser le callback ou décaler les arguments.                       | Corriger la signature et ajouter un test introspection.                                                                                     |
|     High | Sizing min stake            | `custom_stake_amount()` force `stake = max(min_stake, stake)`.                                                                                                                                                                     | Si le minimum exchange dépasse l’unité d’inventaire, le bot augmente le risque au lieu de refuser.              | Retourner `0` ou `None` si `min_stake` dépasse le budget d’une unité. Freqtrade documente que `0`/`None` empêche le trade. ([Freqtrade][1]) |
|     High | Logging des accepted quotes | `custom_entry_price()` peut logger `decision="accept"` après maker-safe local, sans appeler toute `_quote_state_valid()`.  Le canary report a déjà trouvé des accepted quotes stale collector/orderbook/params.                    | Les logs peuvent surestimer la qualité des quotes et polluer la calibration.                                    | Ne jamais logger `accept` avant validation complète.                                                                                        |
|     High | `Alo` dans Freqtrade config | Les tests internes acceptent `time_in_force="Alo"` quand post-only est vérifié.  Mais la doc Freqtrade stable liste la config `order_time_in_force` comme `GTC/FOK/IOC`, avec support exchange-dependent pour PO. ([Freqtrade][2]) | Le runtime Freqtrade peut rejeter `Alo` même si le test unitaire mocké passe.                                   | Ne pas supposer que `Alo` marche via Freqtrade; tester dans le conteneur exact.                                                             |
|   Medium | Callbacks lourds            | La doc Freqtrade dit d’éviter les calculs lourds dans les callbacks; `bot_loop_start()` est appelé à chaque boucle live/dry-run. ([Freqtrade][1])                                                                                  | Les estimateurs synchrones peuvent bloquer le loop, vieillir les quotes et augmenter le risque de stale orders. | Déplacer les estimateurs dans un sidecar; la stratégie lit seulement des snapshots atomiques.                                               |
|   Medium | Stop/risk futures           | Hyperliquid futures supporte stoploss on-exchange en limit selon la doc Freqtrade, et Freqtrade recommande prudence sur stoploss-limit en volatilité. ([Freqtrade][3])                                                             | Un kill switch qui annule les ordres ouverts ne suffit pas toujours à réduire une position déjà remplie.        | Ajouter un chemin d’urgence reduce-only/flatten vérifié, séparé du quoting maker.                                                           |

## Points de conformité Freqtrade à corriger en priorité

### 1) Corriger `custom_stake_amount()`

La signature doit être compatible avec la doc stable actuelle :

```python
def custom_stake_amount(
    self,
    pair: str,
    current_time: datetime,
    current_rate: float,
    proposed_stake: float,
    min_stake: float | None,
    max_stake: float,
    leverage: float,
    entry_tag: str | None,
    side: str,
    **kwargs,
) -> float | None:
    ...
```

Puis changer la logique de min-stake. Aujourd’hui, si `min_stake` est supérieur à l’unité cible, le code peut remonter la taille. Pour un market maker inventaire-borné, c’est le mauvais comportement. Il faut refuser :

```python
one_unit_stake = self.inventory_unit_base * rate
risk_cap = min(proposed, maximum, one_unit_stake)

if min_stake is not None and float(min_stake) > risk_cap:
    self._debug_log_event("stake_rejected", {
        "pair": pair,
        "reason": "min_stake_exceeds_inventory_unit",
        "min_stake": float(min_stake),
        "risk_cap": float(risk_cap),
        "inventory_unit_base": float(self.inventory_unit_base),
        "current_rate": float(rate),
    })
    return 0.0
```

C’est aligné avec la doc Freqtrade : `0` ou `None` empêche le trade, tandis qu’une exception retombe sur `proposed_stake`, ce qui serait dangereux ici. ([Freqtrade][1])

### 2) Ne jamais logger une quote acceptée avant validation complète

La doc Freqtrade précise que `custom_entry_price()` / `custom_exit_price()` sont appelées juste avant placement d’ordre et que `None` ou une valeur invalide retombe sur `proposed_rate`. ([Freqtrade][1]) Votre choix de retourner `proposed_rate` sur certains rejets puis de bloquer dans `confirm_trade_entry/exit()` est donc logique. Mais le bug restant est observabilité/calibration : `custom_entry_price()` peut logger une quote acceptée si `_maker_safe()` passe, même si les params, collector, HJB ou gates live seraient rejetés ensuite par `_quote_state_valid()`. 

Implémentation recommandée :

```python
ok, reason = self._quote_state_valid(pair, "bid", returned_rate, current_time)
if not ok:
    self._log_quote_decision(
        pair=pair,
        symbol=symbol,
        side="bid",
        action="entry",
        decision="reject",
        reason=reason,
        mid_price=mid_price,
        proposed_rate=proposed_rate,
        raw_price=raw_rate,
        rounded_price=returned_rate,
    )
    return proposed_rate

ok, reason = self._maker_safe(pair, "bid", returned_rate)
...
```

Même chose côté ask. Le principe : **`quote_decision=accept` doit signifier “acceptable jusqu’au dernier guard connu”, pas seulement “ne traverse pas le book local”.**

### 3) Garder `confirm_trade_entry/exit()` comme dernier garde-fou, mais léger

La doc Freqtrade dit que `confirm_trade_entry()` et `confirm_trade_exit()` sont les dernières méthodes appelées avant placement d’ordre, et que le timing est critique : pas de calcul lourd ni requête réseau. ([Freqtrade][1]) ([Freqtrade][1]) Votre architecture va dans le bon sens si ces callbacks restent des checks rapides sur état déjà chargé.

Très bon point : vous laissez passer les sorties protégées type stop-loss/emergency/force exit. C’est important car Freqtrade avertit que `confirm_trade_exit()` peut bloquer les stoploss et causer des pertes significatives. ([Freqtrade][1])

### 4) Ne pas utiliser Freqtrade live comme executor post-only tant que `Alo` n’est pas prouvé

La config actuelle est cohérente en dry-run : `GTC` parce que Freqtrade/Hyperliquid PO n’est pas vérifié.  Mais il ne faut pas promouvoir Freqtrade live juste en mettant `order_time_in_force={"entry":"Alo","exit":"Alo"}` : la doc Freqtrade dit que les valeurs config supportées sont `GTC`, `FOK`, `IOC`, et demande de vérifier le support exchange pour les time-in-force. ([Freqtrade][2]) Le repo documente déjà que Freqtrade 2025.4 a rejeté `PO` pour Hyperliquid et que le chemin acceptable est soit preuve native `Alo`, soit executor direct SDK. 

Donc la règle doit être stricte : **Freqtrade = recherche/dry-run jusqu’à preuve runtime exacte. Direct SDK = seul chemin live probable pour `Alo`.**

### 5) Corriger la différence dry-run/backtest vs maker réel

Freqtrade dry-run simule les ordres et ne poste rien à l’exchange; les limit orders remplissent quand le prix atteint le niveau, et les limits très crossing peuvent être convertis en market dans le simulateur. ([Freqtrade][2]) En backtest, les custom prices remplissent si le prix tombe dans le high/low de la candle. ([Freqtrade][1]) Pour du market making, ça ne prouve ni queue position, ni maker-only, ni adverse selection. Votre replay dédié est donc nécessaire; il faut juste le pousser jusqu’au multi-jours avec fills calibrés.

## Plan d’implémentation détaillé

### Phase A — Compatibilité Freqtrade et fail-closed

1. Verrouiller la version de doc cible : votre image est `freqtradeorg/freqtrade:2025.4`; il faut tester contre cette image, pas seulement contre la doc stable actuelle. 
2. Corriger la signature `custom_stake_amount()` avec `leverage`.
3. Ajouter un test introspection des callbacks contre la version installée dans le conteneur : `custom_entry_price`, `custom_exit_price`, `custom_stake_amount`, `confirm_trade_entry`, `confirm_trade_exit`, `adjust_entry_price`, `adjust_exit_price`, `order_filled`.
4. Changer le default strategy `emergency_exit` vers `market`, même si la config l’override, pour éviter qu’un lancement sans config complète garde `emergency_exit: limit`. La doc Freqtrade dit que `order_types` en config override toute la stratégie et doit être complet; elle montre aussi `emergency_exit: market` comme standard. ([Freqtrade][2])
5. Supprimer les imports inutiles qui peuvent casser le runtime, notamment tout import non installé explicitement dans `Dockerfile.technical`.

### Phase B — Pipeline quote propre

Créer une abstraction unique :

```python
@dataclass(frozen=True)
class QuoteCandidate:
    pair: str
    side: Literal["bid", "ask"]
    action: Literal["entry", "exit", "adjust_entry", "adjust_exit"]
    mid: float
    raw_price: float | None
    rounded_price: float | None
    delta_model: float | None
    fee_cushion: float | None
    reason: str
    accepted: bool
```

Puis faire passer **toutes** les quotes par la même séquence :

1. HJB cache présent et frais.
2. Params valides, `status="ok"`, timestamps non stale et non futurs.
3. Collector frais par row timestamp, pas par mtime.
4. Orderbook frais.
5. Fee state valide.
6. Inventaire autorise le côté.
7. Delta HJB fini; `np.inf` = côté désactivé.
8. Prix arrondi au tick dans le sens maker.
9. Prix dans `custom_price_max_distance_ratio`; sinon rejet avant que Freqtrade le clamp silencieusement.
10. Maker-safe local BBO.
11. Seulement là : log `quote_decision accept` et cache quote_id.

### Phase C — Sizing et inventaire

1. `custom_stake_amount()` doit retourner `0.0` si le min exchange dépasse `inventory_unit_base * price`.
2. Ajouter un callback `leverage()` explicite qui retourne `1.0` tant que le système n’a pas une logique de liquidation/margin complète.
3. Refuser les positions short inattendues comme aujourd’hui, mais ajouter un mode “flatten” séparé : un kill switch annule les quotes, puis un executor de risque ferme/reduit la position si l’inventaire dépasse les bornes.
4. Ajouter `max_notional_exposure_usdc`, `max_margin_used_usdc`, `min_liquidation_buffer_usdc` comme guards live, pas seulement métriques de replay.

### Phase D — Post-only/Alo

1. Ne pas débloquer Freqtrade live avec `GTC`.
2. Tenter d’abord une preuve runtime dans le conteneur exact : config `PO` ou `Alo`, `freqtrade list-strategies`, dry-run startup, puis submit minimal testnet si Freqtrade accepte.
3. Si Freqtrade rejette encore `PO/Alo`, utiliser uniquement le direct SDK scaffold. Le repo a déjà un adapter direct qui construit `order_type={"limit":{"tif":"Alo"}}` et un checker qui exige crossing reject/cancel sans fill, passive resting ou maker fill, TIF réel `Alo`, fresh timestamps, et aucune liquidité taker. 
4. Ajouter un `client_order_id`/`cloid` contenant `quote_id`, `side`, `generation`, et `session_id` si Hyperliquid le supporte dans le chemin choisi.
5. Tous les fills doivent réconcilier `quote_id -> order_attempt_accepted -> exchange fill`. Le canary verifier exige déjà cette logique. 

### Phase E — Paramètres et sidecar

1. Sortir les estimateurs du callback Freqtrade. Le collector et les estimateurs doivent tourner comme services séparés : `hl-collector`, `param-estimator`, puis `freqtrade-research`.
2. La stratégie ne doit jamais lancer `get_kappa/get_epsilon/get_lambda` dans le loop live. Elle lit seulement des JSON atomiques déjà validés.
3. Ajouter un champ `not_valid_before` / `window_end` strict, et refuser tout timestamp futur au-delà d’un petit skew, par exemple 5–10 secondes.
4. Conserver le lock `param_update.lock`, mais ajouter un TTL : si le process meurt, le lock stale ne doit pas bloquer indéfiniment; il doit aussi faire fail-closed jusqu’à intervention.

### Phase F — Replay et evidence

1. Rejouer au moins 3–7 jours de données par symbole. Le rapport actuel échoue parce que la couverture est trop faible et qu’il n’y a pas de maker fills. 
2. Calibrer les fills à partir de logs testnet avec `quote_id`, profondeur bps, queue ahead, fill delay, markout 100 ms / 1 s / 5 s / 30 s.
3. Refuser promotion si PnL vient principalement de drift directionnel plutôt que spread capture net fees. Le repo a déjà prévu cette attribution dans le replay. 
4. Tester sensibilité : fee x2, latency x2, kappa/lambda/epsilon perturbés, widened tick, stale data, missing streams.
5. Exiger zéro maintenance-margin breach et une liquidation buffer positive dans toutes les variantes.

### Phase G — Canary live minimal

1. Débloquer seulement `deployment_stage=canary` quand post-only, fee evidence et replay sont `ok=true`, frais et preuves frais dans la fenêtre configurée. Le repo vérifie déjà ces rapports et refuse les rapports stale/missing. 
2. Taille : un seul symbole, `stake_amount=25`, cap notional strict, leverage 1x, `post_only_verified=true`, kill-on-taker-fill, kill-on-unknown-liquidity, kill-on-TIF-mismatch.
3. Sessions : au moins 3 sessions séparées, monitoring manuel loggé, zéro taker fill, fills maker réconciliés au quote_id, fee réel conforme.
4. Production seulement quand `live_canary_report.json` est `ok=true`; le rapport actuel ne l’est pas. 

## Tests précis à ajouter

```python
def test_custom_stake_amount_signature_matches_freqtrade_stable():
    params = list(inspect.signature(Market_Making.custom_stake_amount).parameters)
    assert params[:9] == [
        "self", "pair", "current_time", "current_rate",
        "proposed_stake", "min_stake", "max_stake",
        "leverage", "entry_tag",
    ]
```

```python
def test_stake_returns_zero_when_min_stake_exceeds_inventory_unit():
    bot = make_bot()
    bot.inventory_unit_base = 0.01
    stake = bot.custom_stake_amount(
        "ETH/USDC:USDC",
        datetime.now(timezone.utc),
        current_rate=100.0,
        proposed_stake=25.0,
        min_stake=5.0,
        max_stake=25.0,
        leverage=1.0,
        entry_tag="mm_bid",
        side="long",
    )
    assert stake == 0.0
```

```python
def test_custom_entry_does_not_log_accept_when_params_stale():
    bot = make_bot()
    bot.trading_enabled = True
    bot.kappas["ETH"]["generated_at"] = "2026-01-01T00:00:00Z"
    events = []
    bot._debug_log_event = lambda event, payload: events.append((event, payload))

    returned = bot.custom_entry_price(
        "ETH/USDC:USDC", None, datetime.now(timezone.utc),
        proposed_rate=99.5, entry_tag="mm_bid", side="long",
    )

    assert returned == 99.5
    assert events[0][0] == "quote_decision"
    assert events[0][1]["decision"] == "reject"
    assert events[0][1]["reason"] == "param_stale"
```

```python
def test_future_param_timestamp_rejected():
    bot = make_bot()
    bot.kappas["ETH"]["generated_at"] = "2999-01-01T00:00:00Z"
    assert bot._params_are_valid("ETH/USDC:USDC") == (False, "param_timestamp_future")
```

```python
def test_future_gate_report_timestamp_rejected(tmp_path):
    bot = make_bot()
    path = tmp_path / "fee.json"
    path.write_text(json.dumps({"ok": True, "generated_at": "2999-01-01T00:00:00Z"}))
    status = bot._read_gate_report_status(path)
    assert status["ok"] is False
    assert status["reason"] == "future_generated_at"
```

## Commandes de promotion recommandées

Pour rester cohérent avec l’architecture actuelle :

```bash
python scripts/run_safety_gates.py --include-runtime --markdown-output docs/LAST_SAFETY_GATES.md
```

Puis, après preuves testnet/tiny :

```bash
python scripts/run_safety_gates.py \
  --include-runtime \
  --audit-log-input docs/testnet_mm_debug.jsonl \
  --post-only-crossing-result docs/post_only_crossing_result.json \
  --post-only-passive-result docs/post_only_passive_result.json \
  --max-evidence-age-seconds 3600 \
  --max-canary-event-age-seconds 86400 \
  --replay-acceptance-newest-per-stream 0 \
  --replay-acceptance-max-price-events 0 \
  --replay-acceptance-require-pass \
  --manual-monitoring-ack \
  --markdown-output docs/LAST_SAFETY_GATES.md
```

## Conclusion

Le code est **bien avancé pour un framework de recherche/dry-run** : beaucoup de bugs classiques ont déjà été traités, notamment boundary HJB, fallback `proposed_rate`, inventaire signé, stale data, fee mismatch, et gates de déploiement. Mais les blocages restants sont exactement ceux qui comptent pour un market maker live : **post-only réel, fee réel, fills maker réconciliés, replay multi-jours, canary live**. Le plus gros correctif code immédiat est la compatibilité Freqtrade de `custom_stake_amount()` et le nettoyage du pipeline de logging/validation des quotes; le plus gros correctif architecture est de **ne pas utiliser Freqtrade comme executor live Hyperliquid tant que `Alo` n’est pas prouvé dans le runtime exact**.

[1]: https://www.freqtrade.io/en/stable/strategy-callbacks/ "Strategy Callbacks - Freqtrade"
[2]: https://www.freqtrade.io/en/stable/configuration/ "Configuration - Freqtrade"
[3]: https://www.freqtrade.io/en/stable/stoploss/ "Stoploss - Freqtrade"
