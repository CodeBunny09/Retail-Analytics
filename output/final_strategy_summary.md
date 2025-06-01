
## Key Insights

1. **Three Distinct Store Segments** identified with varying sales patterns.  
   *Segment 0* shows the highest average weekly sales (see bar chart).

2. **Association Rules** reveal strong product affinities, e.g.  
   *frozenset({37}) ⇒ frozenset({49})*  
   with Lift 1.51.  
   Use these for cross‑merchandising and bundle promotions.

3. **Forecast Accuracy**  
   RMSE: 2129776.59, MAPE: 3.68%.  
   The SARIMAX model captures seasonality without explosive errors.

## Strategic Recommendations

| Area | Action |
|------|--------|
| **Inventory** | Increase stock for bundles found in high‑lift rules, especially in Segment 0 stores. |
| **Marketing** | Promote combo deals featuring Dept pairs (e.g., frozenset({37})) online and in‑store. |
| **Personalization** | For shoppers purchasing antecedent items, recommend the consequent in real‑time (email or app push). |
| **Store Layout** | Place frequently co‑occurring Departments nearer to reduce shopper friction. |
| **Forecasting** | Re‑train SARIMAX monthly; feed forecast output to replenishment planning. |

