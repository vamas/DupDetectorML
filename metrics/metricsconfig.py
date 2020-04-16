class MetricsConfig(object):

    def __init__(self, metrics):
        self.metrics = metrics

    def getMetricNames(self, feature_name):
        feature_metrics = {}
        metric_config = self.metrics
        for metric_key in metric_config.keys():
            feature_metrics[metric_key] = metric_config[metric_key] + feature_name
        return feature_metrics

    def getMetrics(self):
        return self.metrics

    
    

    