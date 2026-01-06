# RESIDUALS-as-a-Service: Concept Document

*Captured from conversation, January 2026*

## Context

RESIDUALS generates exhaustive characterizations of decomposition × upsampling method combinations on LiDAR DEMs. The full corpus is ~5TB of prior art, hashed and timestamped for provenance.

**Problem:** The full dataset is too large to deploy as a service, but the *knowledge* it contains could help researchers select optimal methods for their data.

## Service Concept

A lightweight, queryable service that answers: *"Which decomposition × upsampling combination works best for my data?"*

### Architecture Options

#### 1. Signature Lookup (Recommended)

Pre-compute "fingerprints" for each method combination's behavior:
- How it responds to ridges, valleys, edges, noise
- Characteristic residual patterns
- Frequency response profile

**Query flow:**
```
User uploads sample → Extract terrain signature → 
Match against method response signatures → 
Return ranked method recommendations
```

**Storage:** Megabytes (signature vectors only)

#### 2. Decision Model

Train a model on the 5TB corpus:
- Input: Data characteristics (roughness, feature types, noise level)
- Output: Recommended method + confidence

**Deployment:** Just the model weights, no raw data needed

**Query flow:**
```
User describes data characteristics → 
Model inference → 
Return recommendations with explanations
```

#### 3. Representative Gallery

Curated examples showing each method's strengths/weaknesses:
- One canonical input/output pair per method
- Visual comparison grid
- Metadata: best for X, avoid for Y

**Use case:** Educational, method selection guidance

### Integration with Vector Projects

The HNSW architecture from terravector/satvector/etc. could power the signature lookup:

```
RESIDUALS Corpus (5TB)
        ↓
   Characterization
        ↓
Method Signature Index (HNSW)
        ↓
    Query API
        ↓
"Use bilateral × lanczos for your ridge detection"
```

### API Sketch

```python
# Recommend methods for a terrain sample
POST /api/recommend
{
  "sample": <base64 encoded patch>,
  "task": "ridge_detection",  # optional
  "top_k": 5
}

# Response
{
  "recommendations": [
    {
      "decomposition": "bilateral",
      "upsampling": "lanczos", 
      "confidence": 0.92,
      "reason": "Strong edge preservation, good for linear features"
    },
    ...
  ]
}

# Get method details
GET /api/methods/bilateral_lanczos
{
  "name": "bilateral × lanczos",
  "strengths": ["edge preservation", "noise reduction"],
  "weaknesses": ["slow on large patches"],
  "example_output": <url>
}
```

### Pruning Strategy for Deployment

From 5TB archive to deployable service:

1. **Keep:** Method signature vectors (computed, not raw)
2. **Keep:** One representative example per method combination
3. **Keep:** Trained recommendation model weights
4. **Keep:** Hash manifest proving full corpus existed
5. **Archive:** Full 5TB (sacred first dataset, not deployed)

**Estimated deployed size:** 50-100GB for gallery, <1GB for signatures/model

## Related Projects

| Project | Role |
|---------|------|
| RESIDUALS | Source corpus, prior art |
| TerraVector | Signature architecture, HNSW indexing |
| SatVector | Potential satellite data extension |
| AstroVector | Potential astronomy extension |
| CMBVector | Potential cosmology extension |
| FieldVector | Generic 2D field extension |

## Next Steps

1. Complete hash verification of 5TB corpus
2. Publish hash manifest for prior art timestamp
3. Design signature extraction from characterizations
4. Build prototype recommendation API
5. Create representative gallery

---

*This document captures ideas from a productive session. Implementation details TBD.*

