package org.apache.mahout.cf.taste.impl.similarity;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.model.DataModel;

public final class WeightedPearsonCorrelationSimilarity extends AbstractSimilarity {

	public static final int THRESHOLD = 50;
	
	private int threshold;
	
	public WeightedPearsonCorrelationSimilarity(DataModel dataModel) throws TasteException {
		this(dataModel, THRESHOLD);
	}
	
	public WeightedPearsonCorrelationSimilarity(DataModel dataModel, int threshold) throws TasteException {
		this(dataModel, Weighting.UNWEIGHTED, threshold);
	}
	
	public WeightedPearsonCorrelationSimilarity(DataModel dataModel,
			Weighting weighting, int threshold) throws TasteException {
		super(dataModel, weighting, true);
		this.threshold = threshold;
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
	    if(n <= threshold && threshold > 0) {
	    	return (sumXY / denominator) * (n / threshold);
	    } else {
	    	return sumXY / denominator;
	    }
	}

}
