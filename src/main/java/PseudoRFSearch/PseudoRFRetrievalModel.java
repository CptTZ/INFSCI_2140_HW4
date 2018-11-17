package PseudoRFSearch;

import Classes.Document;
import Classes.Query;
import IndexingLucene.MyIndexReader;
import SearchLucene.HW3QRModel;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class PseudoRFRetrievalModel {

    private final MyIndexReader ixreader;
    private final long allContentLength;
    private final HW3QRModel qrm;

    private final double mu = 2000;

    public PseudoRFRetrievalModel(MyIndexReader ixreader) {
        this.ixreader = ixreader;
        this.allContentLength = ixreader.getTotalContentLength();
        this.qrm = new HW3QRModel(ixreader);
        this.qrm.setMu(this.mu);
    }

    /**
     * Search for the topic with pseudo relevance feedback in 2017 spring assignment 4.
     * The returned results (retrieved documents) should be ranked by the score (from the most relevant to the least).
     *
     * @param aQuery The query to be searched for.
     * @param TopN   The maximum number of returned document
     * @param TopK   The count of feedback documents
     * @param alpha  parameter of relevance feedback model
     * @return TopN most relevant document, in List structure
     */
    public List<Document> RetrieveQuery(Query aQuery, int TopN, int TopK, double alpha) throws Exception {
        // this method will return the retrieval result of the given Query, and this result is enhanced with pseudo relevance feedback
        // (1) you should first use the original retrieval model to get TopK documents, which will be regarded as feedback documents
        // (2) implement GetTokenRFScore to get each query token's P(token|feedback model) in feedback documents
        // (3) implement the relevance feedback model for each token: combine the each query token's original retrieval score P(token|document) with its score in feedback documents P(token|feedback model)
        // (4) for each document, use the query likelihood language model to get the whole query's new score, P(Q|document)=P(token_1|document')*P(token_2|document')*...*P(token_n|document')
        double revAlpha = 1 - alpha;

        // sort all retrieved documents from most relevant to least, and return TopN
        List<Document> results = new ArrayList<>(TopN);

        // convert query to terms
        String[] tokens = aQuery.GetQueryContent().split(" ");

        //get P(token|feedback documents)
        HashMap<String, Double> TokenRFScore = GetTokenRFScore(aQuery, TopK);

        // Get posting like in hw3
        HashMap<Integer, HashMap<String, Integer>> postingData = this.qrm.getQueryResult();

        // Relevance feedback model
        for (int docid : postingData.keySet()) {
            HashMap<String, Integer> posting = postingData.get(docid);
            int doclen = this.ixreader.docLength(docid);
            double score = 1.0;

            // Smooth (come from hw3)
            double adjLen = (doclen + this.mu);
            double // (|D|/(|D|+MU)) as l1 and (MU/(|D|+MU)) as r1
                    l1 = 1.0 * doclen / adjLen,
                    r1 = 1.0 * this.mu / adjLen;
            for (String token : tokens) {
                long cf = this.ixreader.CollectionFreq(token);
                if (cf == 0) {
                    // System.err.println("Token " + token + " not appeared");
                    continue;
                }
                long tf = posting.getOrDefault(token, 0);
                double // p(w|D) = l1*(c(w,D)/|D|) + r1*p(w|REF)
                        l2 = 1.0 * tf / doclen,
                        r2 = 1.0 * cf / this.allContentLength;
                // Unigram LM
                score *= alpha * (l1 * l2 + r1 * r2) + revAlpha * TokenRFScore.getOrDefault(token, 0.0);
            }
            results.add(new Document(String.valueOf(docid), ixreader.getDocno(docid), score));
        }

        results.sort((d1, d2) -> d1.score() > d2.score() ? -1 : 1);
        return results.subList(0, TopN);
    }

    /**
     * for each token in the query, you should calculate token's score in feedback documents: P(token|feedback documents)
     * use Dirichlet smoothing
     * save <token, score> in HashMap TokenRFScore, and return it
     *
     * @param aQuery
     * @param TopK
     * @return
     */
    private HashMap<String, Double> GetTokenRFScore(Query aQuery, int TopK) throws Exception {
        HashMap<String, Double> TokenRFScore = new HashMap<>();

        List<Document> originalRes = this.qrm.retrieveQuery(aQuery, TopK);
        HashMap<Integer, HashMap<String, Integer>> p = this.qrm.getQueryResult();

        HashMap<String, Long> pDoc = new HashMap<>();
        long doclen = populatePseudoDoc(pDoc, originalRes, p);

        // Smooth (come from hw3)
        double adjLen = (doclen + this.mu);
        double // (|D|/(|D|+MU)) as l1 and (MU/(|D|+MU)) as r1
                l1 = 1.0 * doclen / adjLen,
                r1 = 1.0 * this.mu / adjLen;
        for (String token : pDoc.keySet()) {
            long cf = this.qrm.getCollectionFreq(token);
            if (cf == 0) {
                // System.err.println("Token " + token + " not appeared");
                continue;
            }
            long tf = pDoc.getOrDefault(token, 0L);
            double // p(w|D) = l1*(c(w,D)/|D|) + r1*p(w|REF)
                    l2 = 1.0 * tf / doclen,
                    r2 = 1.0 * cf / this.allContentLength;
            // Unigram LM
            TokenRFScore.put(token, l1 * l2 + r1 * r2);
        }

        return TokenRFScore;
    }

    private long populatePseudoDoc(HashMap<String, Long> doc, List<Document> orig, HashMap<Integer, HashMap<String, Integer>> postings) throws IOException {
        long totalLen = 0;
        for (Document d : orig) {
            int docId = Integer.valueOf(d.docid());
            HashMap<String, Integer> posting = postings.get(docId);
            if (posting == null) {
                System.err.println("Document not in posting list");
                continue;
            }
            for (String token : posting.keySet()) {
                int tf = posting.getOrDefault(token, 0);
                long old = doc.getOrDefault(token, 0L);
                doc.put(token, old + tf);
            }
            totalLen += this.ixreader.docLength(docId);
        }
        return totalLen;
    }

}
