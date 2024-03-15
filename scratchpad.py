# %%
import sampleGeneration
temp = sampleGeneration.Sample("Square")
temp.showSample()
temp.makeNumpyVersion()
print(temp.sample.shape)

# %%
import sampleGeneration
temp = sampleGeneration.Sample("Ellipse")
temp.showSample()
temp.makeNumpyVersion()
print(temp.sample.shape)

# %%
samples = list()
for i in range(10):
    samples.append(sampleGeneration.Sample("Circle"))
    samples.append(sampleGeneration.Sample("Square"))
# %%
samples[1].showSample()
# %%
