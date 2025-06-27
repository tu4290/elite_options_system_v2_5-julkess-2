-- =====================================================
-- LEGENDARY MOE DATABASE SCHEMAS FOR SUPABASE
-- =====================================================
-- Each MOE gets its own comprehensive database
-- Built for learning, evolution, and domain expertise
-- =====================================================

-- =====================================================
-- 🌊 MARKET REGIME MOE DATABASE
-- =====================================================

-- Core training data for regime analysis
CREATE TABLE market_regime_training (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Market State Data
    regime_type VARCHAR(50) NOT NULL, -- bull, bear, sideways, volatile
    regime_strength DECIMAL(5,3), -- 0.0 to 1.0
    regime_duration INTEGER, -- days in current regime
    transition_probability DECIMAL(5,3), -- probability of regime change
    
    -- Core EOTS Metrics
    vri_2_0 DECIMAL(8,4),
    vri_raw DECIMAL(8,4),
    vri_smoothed DECIMAL(8,4),
    volatility_rank DECIMAL(5,3),
    volatility_percentile DECIMAL(5,3),
    
    -- Market Context
    spy_price DECIMAL(10,2),
    spy_change_pct DECIMAL(6,3),
    spy_volume BIGINT,
    vix_level DECIMAL(6,2),
    vix_change_pct DECIMAL(6,3),
    
    -- Regime Indicators
    trend_strength DECIMAL(5,3),
    momentum_score DECIMAL(8,4),
    mean_reversion_signal DECIMAL(8,4),
    breakout_probability DECIMAL(5,3),
    consolidation_score DECIMAL(5,3),
    
    -- Time-based Features
    hour_of_day INTEGER,
    day_of_week INTEGER,
    days_to_expiry INTEGER,
    is_earnings_week BOOLEAN,
    is_fomc_week BOOLEAN,
    
    -- Outcome Data (for learning)
    actual_regime_next_hour VARCHAR(50),
    actual_regime_next_day VARCHAR(50),
    actual_regime_next_week VARCHAR(50),
    regime_change_occurred BOOLEAN,
    prediction_accuracy DECIMAL(5,3),
    
    -- Metadata
    data_source VARCHAR(100),
    confidence_score DECIMAL(5,3),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Regime pattern recognition
CREATE TABLE regime_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_type VARCHAR(50), -- transition, continuation, reversal
    
    -- Pattern Conditions
    vri_min DECIMAL(8,4),
    vri_max DECIMAL(8,4),
    volatility_condition VARCHAR(50),
    trend_condition VARCHAR(50),
    volume_condition VARCHAR(50),
    
    -- Pattern Outcomes
    success_rate DECIMAL(5,3),
    avg_duration_hours INTEGER,
    avg_move_magnitude DECIMAL(6,3),
    confidence_level DECIMAL(5,3),
    
    -- Learning Metrics
    times_detected INTEGER DEFAULT 0,
    times_correct INTEGER DEFAULT 0,
    last_seen TIMESTAMPTZ,
    pattern_strength DECIMAL(5,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Regime MOE performance tracking
CREATE TABLE regime_moe_performance (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Prediction Metrics
    regime_prediction VARCHAR(50),
    regime_confidence DECIMAL(5,3),
    transition_prediction DECIMAL(5,3),
    actual_outcome VARCHAR(50),
    prediction_correct BOOLEAN,
    
    -- Performance Scores
    accuracy_1h DECIMAL(5,3),
    accuracy_4h DECIMAL(5,3),
    accuracy_1d DECIMAL(5,3),
    accuracy_1w DECIMAL(5,3),
    
    -- Learning Progress
    learning_rate DECIMAL(8,6),
    model_version VARCHAR(20),
    training_iterations INTEGER,
    loss_function_value DECIMAL(10,6),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- 💧 OPTIONS FLOW MOE DATABASE  
-- =====================================================

-- Core training data for options flow analysis
CREATE TABLE options_flow_training (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Core Flow Metrics
    vapi_fa DECIMAL(8,4),
    dwfd DECIMAL(8,4),
    tw_laf DECIMAL(8,4),
    
    -- Custom Flow Metrics
    lwpai DECIMAL(8,4),
    vabai DECIMAL(8,4),
    aofm DECIMAL(8,4),
    lidb DECIMAL(8,4),
    
    -- Market Context
    spy_price DECIMAL(10,2),
    spy_iv DECIMAL(6,3),
    total_volume BIGINT,
    call_volume BIGINT,
    put_volume BIGINT,
    call_put_ratio DECIMAL(6,3),
    
    -- Flow Characteristics
    institutional_flow_score DECIMAL(8,4),
    retail_flow_score DECIMAL(8,4),
    smart_money_indicator DECIMAL(8,4),
    aggressive_flow_pct DECIMAL(5,3),
    passive_flow_pct DECIMAL(5,3),
    
    -- Gamma Metrics
    total_gamma DECIMAL(15,2),
    call_gamma DECIMAL(15,2),
    put_gamma DECIMAL(15,2),
    gamma_imbalance DECIMAL(8,4),
    gex_level DECIMAL(15,2),
    
    -- Strike Analysis
    max_pain DECIMAL(10,2),
    gamma_flip_level DECIMAL(10,2),
    resistance_levels TEXT, -- JSON array of levels
    support_levels TEXT, -- JSON array of levels
    
    -- Time Decay Factors
    avg_dte DECIMAL(6,2),
    theta_exposure DECIMAL(15,2),
    vega_exposure DECIMAL(15,2),
    
    -- Flow Outcomes
    price_move_1h DECIMAL(6,3),
    price_move_4h DECIMAL(6,3),
    price_move_1d DECIMAL(6,3),
    volatility_realized_1d DECIMAL(6,3),
    flow_prediction_accuracy DECIMAL(5,3),
    
    -- Metadata
    data_quality_score DECIMAL(5,3),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Flow pattern recognition
CREATE TABLE flow_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    pattern_category VARCHAR(50), -- institutional, retail, gamma, squeeze
    
    -- Pattern Conditions
    vapi_fa_condition VARCHAR(100),
    dwfd_condition VARCHAR(100),
    volume_condition VARCHAR(100),
    gamma_condition VARCHAR(100),
    
    -- Pattern Characteristics
    typical_duration_hours INTEGER,
    avg_price_impact DECIMAL(6,3),
    success_probability DECIMAL(5,3),
    risk_reward_ratio DECIMAL(6,3),
    
    -- Learning Metrics
    detection_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    pattern_evolution_score DECIMAL(5,3),
    last_detected TIMESTAMPTZ,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Flow MOE performance tracking
CREATE TABLE flow_moe_performance (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Flow Predictions
    flow_direction_prediction VARCHAR(20), -- bullish, bearish, neutral
    flow_magnitude_prediction DECIMAL(8,4),
    flow_confidence DECIMAL(5,3),
    
    -- Actual Outcomes
    actual_flow_direction VARCHAR(20),
    actual_flow_magnitude DECIMAL(8,4),
    prediction_accuracy DECIMAL(5,3),
    
    -- Performance Metrics
    directional_accuracy DECIMAL(5,3),
    magnitude_accuracy DECIMAL(5,3),
    timing_accuracy DECIMAL(5,3),
    overall_score DECIMAL(5,3),
    
    -- Learning Progress
    model_complexity_score DECIMAL(5,3),
    feature_importance TEXT, -- JSON of feature weights
    learning_velocity DECIMAL(8,6),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- 🧠 MARKET INTELLIGENCE MOE DATABASE
-- =====================================================

-- Core training data for market intelligence
CREATE TABLE intelligence_training (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Intelligence Metrics
    mspi DECIMAL(8,4),
    aofm DECIMAL(8,4),
    sentiment_score DECIMAL(8,4),
    momentum_intelligence DECIMAL(8,4),
    
    -- Market Psychology
    fear_greed_index DECIMAL(5,2),
    put_call_ratio DECIMAL(6,3),
    vix_term_structure DECIMAL(6,3),
    skew_indicator DECIMAL(6,3),
    
    -- News & Sentiment
    news_sentiment_score DECIMAL(8,4),
    social_sentiment_score DECIMAL(8,4),
    analyst_sentiment_score DECIMAL(8,4),
    insider_activity_score DECIMAL(8,4),
    
    -- Technical Intelligence
    technical_score DECIMAL(8,4),
    momentum_score DECIMAL(8,4),
    mean_reversion_score DECIMAL(8,4),
    breakout_score DECIMAL(8,4),
    
    -- Cross-Asset Intelligence
    bond_equity_correlation DECIMAL(8,4),
    dollar_strength_impact DECIMAL(8,4),
    commodity_correlation DECIMAL(8,4),
    crypto_correlation DECIMAL(8,4),
    
    -- Sector Intelligence
    sector_rotation_score DECIMAL(8,4),
    growth_value_ratio DECIMAL(8,4),
    large_small_cap_ratio DECIMAL(8,4),
    defensive_cyclical_ratio DECIMAL(8,4),
    
    -- Economic Intelligence
    economic_surprise_index DECIMAL(8,4),
    earnings_revision_trend DECIMAL(8,4),
    credit_spread_indicator DECIMAL(8,4),
    yield_curve_signal DECIMAL(8,4),
    
    -- Contrarian Indicators
    contrarian_signal_strength DECIMAL(8,4),
    extreme_sentiment_flag BOOLEAN,
    reversal_probability DECIMAL(5,3),
    
    -- Intelligence Outcomes
    intelligence_accuracy_1d DECIMAL(5,3),
    intelligence_accuracy_1w DECIMAL(5,3),
    signal_strength_validation DECIMAL(5,3),
    
    -- Metadata
    intelligence_confidence DECIMAL(5,3),
    data_freshness_score DECIMAL(5,3),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Intelligence pattern recognition
CREATE TABLE intelligence_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pattern_name VARCHAR(100) NOT NULL,
    intelligence_type VARCHAR(50), -- sentiment, momentum, contrarian, cross_asset
    
    -- Pattern Triggers
    trigger_conditions TEXT, -- JSON of conditions
    confirmation_signals TEXT, -- JSON of confirmations
    invalidation_signals TEXT, -- JSON of invalidations
    
    -- Pattern Characteristics
    typical_development_time INTEGER, -- hours
    avg_success_rate DECIMAL(5,3),
    avg_magnitude DECIMAL(6,3),
    risk_level VARCHAR(20),
    
    -- Intelligence Evolution
    pattern_maturity_score DECIMAL(5,3),
    adaptation_rate DECIMAL(8,6),
    market_regime_dependency TEXT, -- JSON
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Intelligence MOE performance tracking
CREATE TABLE intelligence_moe_performance (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Intelligence Predictions
    intelligence_signal VARCHAR(50),
    signal_strength DECIMAL(5,3),
    conviction_level DECIMAL(5,3),
    time_horizon VARCHAR(20),
    
    -- Prediction Validation
    signal_materialized BOOLEAN,
    timing_accuracy DECIMAL(5,3),
    magnitude_accuracy DECIMAL(5,3),
    direction_accuracy DECIMAL(5,3),
    
    -- Learning Metrics
    pattern_recognition_score DECIMAL(5,3),
    adaptation_speed DECIMAL(8,6),
    cross_validation_score DECIMAL(5,3),
    ensemble_contribution DECIMAL(5,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- 🎯 META-ORCHESTRATOR MOE DATABASE
-- =====================================================

-- Cross-expert consensus and decision synthesis
CREATE TABLE orchestrator_decisions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Expert Inputs
    regime_expert_signal VARCHAR(50),
    regime_expert_confidence DECIMAL(5,3),
    flow_expert_signal VARCHAR(50),
    flow_expert_confidence DECIMAL(5,3),
    intelligence_expert_signal VARCHAR(50),
    intelligence_expert_confidence DECIMAL(5,3),
    
    -- Consensus Analysis
    expert_agreement_score DECIMAL(5,3),
    signal_convergence_score DECIMAL(5,3),
    confidence_weighted_average DECIMAL(5,3),
    conflict_resolution_method VARCHAR(50),
    
    -- Meta Decision
    final_signal VARCHAR(50),
    final_confidence DECIMAL(5,3),
    decision_rationale TEXT,
    risk_assessment VARCHAR(50),
    
    -- Context Weighting
    market_regime_weight DECIMAL(5,3),
    flow_analysis_weight DECIMAL(5,3),
    intelligence_weight DECIMAL(5,3),
    historical_performance_weight DECIMAL(5,3),
    
    -- Decision Outcomes
    decision_accuracy DECIMAL(5,3),
    outcome_timing VARCHAR(20),
    outcome_magnitude DECIMAL(6,3),
    lessons_learned TEXT,
    
    -- Meta Learning
    orchestration_effectiveness DECIMAL(5,3),
    expert_reliability_scores TEXT, -- JSON
    dynamic_weighting_performance DECIMAL(5,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Expert communication and collaboration
CREATE TABLE expert_communication (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Communication Metadata
    from_expert VARCHAR(50) NOT NULL,
    to_expert VARCHAR(50), -- NULL for broadcast
    message_type VARCHAR(50), -- signal, question, correction, learning
    priority_level INTEGER, -- 1-5
    
    -- Message Content
    message_content TEXT,
    data_payload TEXT, -- JSON
    confidence_level DECIMAL(5,3),
    
    -- Communication Effectiveness
    message_acknowledged BOOLEAN DEFAULT FALSE,
    response_received BOOLEAN DEFAULT FALSE,
    influence_score DECIMAL(5,3),
    learning_impact DECIMAL(5,3),
    
    -- Cross-Expert Learning
    knowledge_transfer_score DECIMAL(5,3),
    collaboration_effectiveness DECIMAL(5,3),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- System-wide performance tracking
CREATE TABLE system_performance (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    
    -- Individual Expert Performance
    regime_expert_accuracy DECIMAL(5,3),
    flow_expert_accuracy DECIMAL(5,3),
    intelligence_expert_accuracy DECIMAL(5,3),
    orchestrator_accuracy DECIMAL(5,3),
    
    -- System-Wide Metrics
    overall_system_accuracy DECIMAL(5,3),
    expert_consensus_rate DECIMAL(5,3),
    decision_latency_ms INTEGER,
    system_confidence DECIMAL(5,3),
    
    -- Learning Progress
    collective_learning_rate DECIMAL(8,6),
    knowledge_base_size INTEGER,
    pattern_recognition_improvement DECIMAL(5,3),
    adaptation_speed DECIMAL(8,6),
    
    -- Resource Utilization
    computational_load DECIMAL(5,3),
    memory_utilization DECIMAL(5,3),
    data_processing_speed DECIMAL(10,2),
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- =====================================================
-- INDEXES FOR PERFORMANCE
-- =====================================================

-- Market Regime MOE Indexes
CREATE INDEX idx_regime_training_timestamp ON market_regime_training(timestamp);
CREATE INDEX idx_regime_training_regime_type ON market_regime_training(regime_type);
CREATE INDEX idx_regime_patterns_success_rate ON regime_patterns(success_rate DESC);

-- Options Flow MOE Indexes  
CREATE INDEX idx_flow_training_timestamp ON options_flow_training(timestamp);
CREATE INDEX idx_flow_training_vapi_fa ON options_flow_training(vapi_fa);
CREATE INDEX idx_flow_patterns_success_probability ON flow_patterns(success_probability DESC);

-- Intelligence MOE Indexes
CREATE INDEX idx_intelligence_training_timestamp ON intelligence_training(timestamp);
CREATE INDEX idx_intelligence_training_mspi ON intelligence_training(mspi);
CREATE INDEX idx_intelligence_patterns_success_rate ON intelligence_patterns(avg_success_rate DESC);

-- Orchestrator Indexes
CREATE INDEX idx_orchestrator_decisions_timestamp ON orchestrator_decisions(timestamp);
CREATE INDEX idx_orchestrator_decisions_accuracy ON orchestrator_decisions(decision_accuracy DESC);
CREATE INDEX idx_expert_communication_timestamp ON expert_communication(timestamp);
CREATE INDEX idx_system_performance_timestamp ON system_performance(timestamp);

-- =====================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE market_regime_training ENABLE ROW LEVEL SECURITY;
ALTER TABLE regime_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE regime_moe_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE options_flow_training ENABLE ROW LEVEL SECURITY;
ALTER TABLE flow_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE flow_moe_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence_training ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE intelligence_moe_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE orchestrator_decisions ENABLE ROW LEVEL SECURITY;
ALTER TABLE expert_communication ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_performance ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your auth setup)
-- Example: Allow authenticated users to read/write their own data
-- CREATE POLICY "Users can access their own data" ON market_regime_training
--   FOR ALL USING (auth.uid() = user_id);

-- =====================================================
-- FUNCTIONS FOR DATA ANALYSIS
-- =====================================================

-- Function to calculate MOE learning progress
CREATE OR REPLACE FUNCTION calculate_moe_learning_progress(
    expert_name TEXT,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    accuracy_trend DECIMAL(5,3),
    learning_velocity DECIMAL(8,6),
    pattern_discovery_rate DECIMAL(5,3),
    overall_improvement DECIMAL(5,3)
) AS $$
BEGIN
    -- Implementation would calculate learning metrics
    -- This is a placeholder for the actual function
    RETURN QUERY
    SELECT 
        0.85::DECIMAL(5,3) as accuracy_trend,
        0.001234::DECIMAL(8,6) as learning_velocity,
        0.15::DECIMAL(5,3) as pattern_discovery_rate,
        0.12::DECIMAL(5,3) as overall_improvement;
END;
$$ LANGUAGE plpgsql;

-- Function to get expert consensus strength
CREATE OR REPLACE FUNCTION get_expert_consensus_strength(
    timestamp_start TIMESTAMPTZ,
    timestamp_end TIMESTAMPTZ
)
RETURNS DECIMAL(5,3) AS $$
DECLARE
    consensus_strength DECIMAL(5,3);
BEGIN
    SELECT AVG(expert_agreement_score)
    INTO consensus_strength
    FROM orchestrator_decisions
    WHERE timestamp BETWEEN timestamp_start AND timestamp_end;
    
    RETURN COALESCE(consensus_strength, 0.0);
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update triggers to relevant tables
CREATE TRIGGER update_regime_training_updated_at
    BEFORE UPDATE ON market_regime_training
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_flow_training_updated_at
    BEFORE UPDATE ON options_flow_training
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_intelligence_training_updated_at
    BEFORE UPDATE ON intelligence_training
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VIEWS FOR EASY DATA ACCESS
-- =====================================================

-- Latest expert performance view
CREATE VIEW latest_expert_performance AS
SELECT 
    'regime' as expert_type,
    accuracy_1d as daily_accuracy,
    accuracy_1w as weekly_accuracy,
    learning_rate,
    created_at
FROM regime_moe_performance
WHERE created_at >= NOW() - INTERVAL '24 hours'
UNION ALL
SELECT 
    'flow' as expert_type,
    directional_accuracy as daily_accuracy,
    overall_score as weekly_accuracy,
    learning_velocity as learning_rate,
    created_at
FROM flow_moe_performance
WHERE created_at >= NOW() - INTERVAL '24 hours'
UNION ALL
SELECT 
    'intelligence' as expert_type,
    direction_accuracy as daily_accuracy,
    cross_validation_score as weekly_accuracy,
    adaptation_speed as learning_rate,
    created_at
FROM intelligence_moe_performance
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- System health dashboard view
CREATE VIEW system_health_dashboard AS
SELECT 
    sp.timestamp,
    sp.overall_system_accuracy,
    sp.expert_consensus_rate,
    sp.decision_latency_ms,
    sp.system_confidence,
    sp.collective_learning_rate,
    COUNT(ec.id) as expert_communications_count
FROM system_performance sp
LEFT JOIN expert_communication ec ON ec.timestamp >= sp.timestamp - INTERVAL '1 hour'
    AND ec.timestamp <= sp.timestamp
WHERE sp.timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY sp.timestamp, sp.overall_system_accuracy, sp.expert_consensus_rate, 
         sp.decision_latency_ms, sp.system_confidence, sp.collective_learning_rate
ORDER BY sp.timestamp DESC;

-- =====================================================
-- COMMENTS AND DOCUMENTATION
-- =====================================================

COMMENT ON TABLE market_regime_training IS 'Training data for Market Regime MOE - focuses on regime detection, transitions, and market state analysis';
COMMENT ON TABLE options_flow_training IS 'Training data for Options Flow MOE - focuses on institutional flow, gamma dynamics, and options market structure';
COMMENT ON TABLE intelligence_training IS 'Training data for Market Intelligence MOE - focuses on sentiment, momentum, and cross-asset intelligence';
COMMENT ON TABLE orchestrator_decisions IS 'Meta-Orchestrator decision log - tracks expert consensus, conflict resolution, and final decisions';
COMMENT ON TABLE expert_communication IS 'Inter-expert communication log - tracks knowledge sharing and collaboration between MOEs';
COMMENT ON TABLE system_performance IS 'System-wide performance metrics - tracks overall MOE ecosystem health and learning progress';

