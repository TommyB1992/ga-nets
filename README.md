# GA-Nets

**Modular and dynamic neural network framework designed for integration with genetic algorithms.**

GA-Nets è una libreria Python progettata per costruire reti neurali flessibili e personalizzabili, con un'architettura pensata per l'integrazione diretta con algoritmi genetici o evolutivi. Supporta architetture **feedforward**, **ricorrenti**, **sparse**, e **non-standard**, lasciando pieno controllo sulla topologia e sulle funzioni di attivazione/aggregazione.

---

## 🔍 Perché usare GA-Nets?

<img src="https://i.gyazo.com/27e8003df60dbbd21e240a53f8ec093a.png" width="33%"/><img src="https://i.gyazo.com/5325ca9217dbca3151a891739548a01d.png" width="33%"/><img src="https://i.gyazo.com/f566d2364af43dd3a78c8926ed204a51.png" width="33%"/>

- ✅ **Modularità completa**: ogni neurone, connessione e rete è completamente configurabile e combinabile. La libreria permette la creazione di topologie difficilmente ottenibili, se non totalmente impossibili, con l'utilizzo di altre. 
- 🔁 **Compatibilità con algoritmi genetici**: le reti possono essere rappresentate, mutabili ed evolute in strutture adatte a GAs.
- 🧠 **Supporto per topologie arbitrarie**: reti dense, ricorrenti, sparse, strutture non tradizionali.
- 🧪 **Semplicità di test ed estensione**: pensato per la ricerca, l'esplorazione e l'evoluzione automatica di architetture.

---

## Stato del progetto

### TODO: Reti Neurali da Implementare

- [x] Feedforward Neural Network (FFN)
- [x] Recurrent Neural Network (RNN)
- [ ] Long short-term memory (LSTM)
- [ ] Gated recurrent units (GRU)
- [ ] Convolutional Neural Network (CNN)
- [ ] Residual Neural Network (ResNet)
- [ ] Autoencoder
- [ ] Generative Adversarial Network (GAN)
- [ ] Transformer
- [ ] Recursive Neural Network (RNN)
- [ ] Modular / Compositional Networks
- [ ] Spiking Neural Networks (SNN)
- [ ] Graph Neural Networks (GNN)

---

### TODO: Sviluppo & Refactoring

- [ ] Refactoring ga_nets/layer.py
- [ ] Distingui la logica (SOLID) fra connessione tra neuroni eliminandola dalla classe Network (add_synapse) e creando una classe apposita
- [ ] Aggiungere la documentazione
- [ ] Riscrivere tutto in inglese 
- [ ] Document all functions with docstring
- [ ] Add tests and examples
- [ ] Set hard typing with typeguard
- [ ] Insert softmax activation
- [ ] Publish library on PyPI

---

## 🛠️ Installazione

È richiesto Python **>= 3.11**.

Clona il repository e installa con `pip`:

```bash
git clone https://github.com/tommyb1992/ga-nets.git
cd ga-nets
pip install .
```

Oppure installa direttamente da GitHub (solo in lettura):

```bash
pip install git+https://github.com/tommyb1992/ga-nets.git
```

---

## 🚀 Esempio di utilizzo

```python
import random

from ga_nets.nets.ffw import FeedForwardNet, FFWNeuron
from ga_nets.activations import sigmoid
from ga_nets.custom_types import NeuronType

# Create the container
net = FeedForwardNet()

# Create the nodes
input_node = FFWNeuron(neuron_type=NeuronType.INPUT)
hidden_node = FFWNeuron(
    neuron_type=NeuronType.HIDDEN,
    bias=random.random(),
    aggregation=sum,
    activation=sigmoid
)
output_node = FFWNeuron(neuron_type=NeuronType.OUTPUT)

# Add nodes to the network
net.add_neuron(input_node)
net.add_neuron(hidden_node)
net.add_neuron(output_node)

# Connect nodes:
#   input_node -> hidden_node -> output_node
net.add_synapse(input_node, hidden_node, weight=random.random())
net.add_synapse(hidden_node, output_node, weight=random.random())

# See the result
features = [[0], [1]]
for feature in features:
    print(f"{feature} -> {net.activate(feature)}")
```

---

## 📜 Licenza

Questo progetto è distribuito sotto licenza **MIT**.  
Copyright © 2025  
**Tomas Bartoli**  
Email: [tomasbartoli1992@gmail.com](mailto:tomasbartoli1992@gmail.com)

Consulta il file [LICENSE](./LICENSE) per dettagli.

---

## 🌐 Link utili

- [Homepage](https://github.com/tommyb1992/ga-nets)
- [Bug Tracker](https://github.com/tommyb1992/ga-nets/issues)
