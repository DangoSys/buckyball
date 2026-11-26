"""Independent integer, time-step model for the configured 4x16 SYNtzulu network."""
from pathlib import Path
import re

def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value

def decay(value, factor):
    product = value * factor
    return signed((abs(product) >> 12) * (-1 if product < 0 else 1),16)

class Network:
    def __init__(self, weights):
        self.weights = weights
        self.voltages = [[[0]*8 for _ in range(2)] for _ in range(4)]

    def step(self, spikes):
        self.layers = []
        for layer in range(4):
            fired = [[False]*8 for _ in range(2)]
            for lane in range(2):
                for neuron in range(8):
                    current = 0
                    for source in range(16):
                        if spikes >> source & 1:
                            word = self.weights[32*layer+4*neuron+source//4]
                            byte = 4*lane + ((source%4+2)%4)
                            current += signed(word >> (8*byte),8)
                    # Accumulator output is 18 bits, then the LIF input is truncated to 16.
                    stimulus = signed(signed(current,18),16)
                    old = self.voltages[layer][lane][neuron]
                    value = signed(decay(old,(3681,3672,3681,3672)[layer])+stimulus,16)
                    fired[lane][neuron] = value >= (1024 if layer==3 else 1)
                    self.voltages[layer][lane][neuron] = 0 if fired[lane][neuron] else value
            self.layers.append([list(v) for v in self.voltages[layer]])
            spikes = 0
            for group in range(4):
                for lane in range(2):
                    for parity in range(2):
                        if fired[lane][2*group+parity]:
                            spikes |= 1 << (4*group+3-lane-2*parity)
        words = [((self.voltages[3][1][i]&65535)<<16)|(self.voltages[3][0][i]&65535) for i in range(8)]
        return words+[spikes]

def sequence(seed=0x82519437):
    rng=seed
    weights=[]
    for _ in range(128):
        rng=(rng*1664525+1013904223)&0xffffffff
        word=rng<<32
        rng=(rng*1664525+1013904223)&0xffffffff
        weights.append(word|rng)
    due=[0]*68
    frames=[]
    for t in range(68):
        rng=(rng*1664525+1013904223)&0xffffffff
        frame=(rng>>8)&65535 if t<64 else 0
        delay=(rng>>29)&3
        due[t+(delay if t<64 else 0)] |= frame
        frames.append(due[t])
    return weights,frames

if __name__=='__main__':
    import argparse
    import json
    ap=argparse.ArgumentParser()
    ap.add_argument("--logs",type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args()
    root=args.logs
    summary=[]
    for seed in (0x82519437,1,0xdeadbeef):
        weights,frames=sequence(seed)
        net=Network(weights)
        expected=[]
        states=[]
        for frame in frames:
            expected.extend(net.step(frame))
            states.append(net.layers)
        for mode in (0,1):
            name=f'final-random-{mode}.log' if seed==0x82519437 else f'math-{seed}-{mode}.log'
            text=(root/name).read_text()
            assert 'SCU hart0 exit=0' in text,name
            observed=[int(v,16) for v in re.findall(r'^TRACE\n([0-9a-f]{16})$',text,re.M)]
            assert len(observed)==len(expected),(name,len(observed))
            for i,(want,got) in enumerate(zip(expected,observed)):
                assert want==got,(name,'step',i//9,'word',i%9,hex(want),hex(got))
            summary.append(dict(seed=hex(seed),mode=mode,steps=len(frames),
                                voltages=16*len(frames),spike_masks=len(frames),mismatches=0))
        if seed==0x82519437:
            rows=re.findall(r'^NEURON (\d+) (\d+) (\d+) (-?\d+) (-?\d+)$',(root/'math-layers.log').read_text(),re.M)
            assert len(rows)==68*4*8,len(rows)
            for step,layer,neuron,a,b in rows:
                want=states[int(step)][int(layer)]
                assert [int(a),int(b)]==[want[0][int(neuron)],want[1][int(neuron)]],(step,layer,neuron,a,b)
    report=dict(whole_soc_cases=summary,standalone_all_layer_voltages=68*4*16)
    (root/'math-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
