# ML Homework

This README documents the reproduction of the LightGCL model in the ReChorus framework and its performance compared to BPRMF and LightGCN models on various datasets.

## Running the LightGCL Model
- The LightGCL model is located at: ReChorus-master/src/models/general/LightGCL.py.
- To run the LightGCL model, follow these steps:

1. Navigate to the `ReChorus-master` directory.
2. running `pip install -r requirements.txt`.
3. running `python src/main.py --model_name LightGCL`.
   
## Models

- BPRMF
- LightGCN
- LightGCL

## Datasets

- Grocery_and_Gourmet_Food
- MIND_Large(Use small version)
- MovieLens_1M

## Metrics

- HR@20
- NDCG@20
- HR@40
- NDCG@40

## Performance

### 
![Grocery Performance](result/Grocery_and_Gourmet_Food.png)

### 
![MIND Performance](result/MIND_Large.png)

### 
![MovieLens Performance](result/MovieLens_1M.png)

## Metrics Data

### Grocery_and_Gourmet_Food

| Model   | HR@20    | NDCG@20  | HR@40    | NDCG@40  |
|---------|----------|----------|----------|----------|
| BPRMF   | 0.5304   | 0.2801   | 0.6783   | 0.3102   |
| LightGCN| 0.6132   | 0.3266   | 0.7619   | 0.3569   |
| LightGCL| 0.6243   | 0.3374   | 0.7679   | 0.3666   |

### MIND_Large

| Model   | HR@20    | NDCG@20  | HR@40    | NDCG@40  |
|---------|----------|----------|----------|----------|
| BPRMF   | 0.1980   | 0.1211   | 0.4667   | 0.1556   |
| LightGCN| 0.2980   | 0.1196   | 0.4716   | 0.1549   |
| LightGCL| 0.3294   | 0.1426   | 0.5422   | 0.1858   |

### MovieLens_1M

| Model   | HR@20    | NDCG@20  | HR@40    | NDCG@40  |
|---------|----------|----------|----------|----------|
| BPRMF   | 0.7495   | 0.3625   | 0.9061   | 0.3948   |
| LightGCN| 0.7216   | 0.3486   | 0.8914   | 0.3836   |
| LightGCL| 0.6670   | 0.3128   | 0.8344   | 0.3474   |


