# active_learning

Python 3.6

```
cd new
bash make_venv.sh
source venv/bin/activate
```

From ipython3, try:
`run strategies/simulator`

## To run on the server:
```
git clone https://github.com/rajisme/active_learning.git
cd active_learning/new
bash setup-server.sh
sudo apt-get install libsuitesparse-dev
(for now): copy install.R into R console
source venv/bin/activate
```
### To zip results:
```
tar -czf <dataset_name>.tar.gz data/<dataset_name>
```
