package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;
import java.util.concurrent.Callable;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.google.common.base.Preconditions;

public final class PIPSimilarity implements UserSimilarity {

	private final DataModel dataModel;
	private final FastByIDMap<RunningAverage> itemAverages;
	private final ReadWriteLock buildAveragesLock;
	private final RefreshHelper refreshHelper;
	
	
	public PIPSimilarity(DataModel dataModel) throws TasteException {
		this.dataModel = Preconditions.checkNotNull(dataModel);
		this.itemAverages = new FastByIDMap<>();
	    this.buildAveragesLock = new ReentrantReadWriteLock();
	    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
	      @Override
	      public Object call() throws TasteException {
	        buildAverageDiffs();
	        return null;
	      }
	    });
	    refreshHelper.addDependency(dataModel);
	    buildAverageDiffs();
	}
	
	private void buildAverageDiffs() throws TasteException {
		try {
			buildAveragesLock.writeLock().lock();
			LongPrimitiveIterator it = dataModel.getUserIDs();
			while (it.hasNext()) {
				PreferenceArray prefs = dataModel.getPreferencesFromUser(it
						.nextLong());
				int size = prefs.length();
				for (int i = 0; i < size; i++) {
					long itemID = prefs.getItemID(i);
					RunningAverage average = itemAverages.get(itemID);
					if (average == null) {
						average = new FullRunningAverage();
						itemAverages.put(itemID, average);
					}
					average.addDatum(prefs.getValue(i));
				}
			}
		} finally {
			buildAveragesLock.writeLock().unlock();
		}
	}

	@Override
	public double userSimilarity(long userID1, long userID2)
			throws TasteException {
	    PreferenceArray xPrefs = dataModel.getPreferencesFromUser(userID1);
	    PreferenceArray yPrefs = dataModel.getPreferencesFromUser(userID2);
	    int xLength = xPrefs.length();
	    int yLength = yPrefs.length();
	    
	    if (xLength == 0 || yLength == 0) {
	      return Double.NaN;
	    }
	    
	    long xIndex = xPrefs.getItemID(0);
	    long yIndex = yPrefs.getItemID(0);
	    int xPrefIndex = 0;
	    int yPrefIndex = 0;
	    
	    double sumPIP = 0.0;
	    
	    while (true) {
	      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
	      if (compare == 0) {
	        double x;
	        double y;
	        x = xPrefs.getValue(xPrefIndex);
	        y = yPrefs.getValue(yPrefIndex);
	        sumPIP = getProximity(x, y) * getImpact(x, y) * getPopularity(x, y, getItemAverage(xIndex));
	      }
	      if (compare <= 0) {
	        if (++xPrefIndex >= xLength) {
	          break;
	        } else {
	          xIndex = xPrefs.getItemID(xPrefIndex);
	        }
	      }
	      if (compare >= 0) {
	        if (++yPrefIndex >= yLength) {
	          break;
	        } else {
	          yIndex = yPrefs.getItemID(yPrefIndex);
	        }
	      }
	    }
	    return sumPIP;
	}
  
	@Override
	public final void setPreferenceInferrer(PreferenceInferrer inferrer) {
		throw new UnsupportedOperationException();
    }
	
	@Override
	public final void refresh(Collection<Refreshable> alreadyRefreshed) {
		alreadyRefreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
	    RefreshHelper.maybeRefresh(alreadyRefreshed, dataModel);
	}
	
	private boolean isAgreement(double x, double y) {
		float maxPref = dataModel.getMaxPreference();
		float minPref = dataModel.getMinPreference();
		double medianPref = (minPref + maxPref) / 2;
		if((x > medianPref && y < medianPref) || (x < medianPref && y > medianPref))
			return false;
		else
			return true;
	}
	
	private double getProximity(double x, double y) {
		float maxPref = dataModel.getMaxPreference();
		float minPref = dataModel.getMinPreference();
		if(isAgreement(x, y)) {
			return 2 * (maxPref - minPref) + 1 - Math.abs(x -y);
		}else{
			return 2 * (maxPref - minPref) + 1 - 2 * Math.abs(x -y);
		}
	}
	
	private double getImpact(double x, double y) {
		float maxPref = dataModel.getMaxPreference();
		float minPref = dataModel.getMinPreference();
		double medianPref = (minPref + maxPref) / 2;
		if(isAgreement(x, y)) {
			return (Math.abs(x - medianPref) + 1) * (Math.abs(y - medianPref) + 1);
		}else{
			return 1 / ((Math.abs(x - medianPref) + 1) * (Math.abs(y - medianPref) + 1));
		}
	}
	
	private double getPopularity(double x, double y, double average) {
		if((x > average && y > average) || (x < average && y < average)) {
			return 1 + Math.pow(((x + y) / 2 - average), 2);
		}else{
			return 1;
		}
	}
	
	private float getItemAverage(long itemID) {
		buildAveragesLock.readLock().lock();
		try {
			RunningAverage average = itemAverages.get(itemID);
			return average == null ? Float.NaN : (float) average.getAverage();
		} finally {
			buildAveragesLock.readLock().unlock();
		}
	}
}
