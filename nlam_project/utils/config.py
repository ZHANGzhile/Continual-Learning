class Config:
    # 模型参数
    INPUT_DIM = 512
    HIDDEN_DIM = 512
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    # Memory Block参数
    SHORT_MEMORY_LAYERS = 2
    MEDIUM_MEMORY_LAYERS = 4
    LONG_MEMORY_LAYERS = 6
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 100
    WARMUP_STEPS = 4000
    
    # 数据参数
    MAX_SEQ_LENGTH = 512
    VOCAB_SIZE = 30000
    
    # 数据参数
    DATA_PATH = "data/raw"
    PROCESSED_DATA_PATH = "data/processed"
    
    # 优化器参数
    BETA1 = 0.9
    BETA2 = 0.999
    WEIGHT_DECAY = 0.01
    
    # 其他参数
    SEED = 42
    DEVICE = "cuda" 