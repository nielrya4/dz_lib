class Grain:
    def __init__(self, age: float, uncertainty: float):
        self.age = age
        self.uncertainty = uncertainty

class Sample:
    def __init__(self, name: str, grains: [Grain]):
        self.name = name
        self.grains = grains

class SampleSheet:
    def __init__(self, name: str, samples: [Sample]):
        self.name = name
        self.samples = samples