package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;

public final class SigmoidPearsonCorrelationSimilarity extends AbstractSimilarity {

	public SigmoidPearsonCorrelationSimilarity(DataModel dataModel) throws TasteException {
		this(dataModel, Weighting.UNWEIGHTED);
	}
	
	public SigmoidPearsonCorrelationSimilarity(DataModel dataModel,
			Weighting weighting) throws TasteException {
		super(dataModel, weighting, true);
	}

	@Override
	double computeResult(int n, double sumXY, double sumX2, double sumY2,
			double sumXYdiff2) {
		if (n == 0) {
	      return Double.NaN;
	    }
	    double denominator = Math.sqrt(sumX2) * Math.sqrt(sumY2);
	    if (denominator == 0.0) {
	      return Double.NaN;
	    }
	    return (sumXY / denominator) * (1 / (1 + Math.exp(-(n / 2))));
	}
}
