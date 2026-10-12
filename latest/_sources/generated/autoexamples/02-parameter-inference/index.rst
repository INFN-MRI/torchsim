

.. _sphx_glr_generated_autoexamples_02-parameter-inference:

Applications: parameter mapping
-------------------------------

Four ways to turn a measured signal into T1, T2 and M0 maps, all built from
the same simulator and compared on the same BrainWeb slice: dictionary
matching, a lookup table, nonlinear least squares and PERK.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="You map T1, T2 and proton density in a brain slice from a magnetic resonance fingerprinting (MRF) train by dictionary matching, then reduce the cost of the match in two independent ways: by working in the low-rank subspace spanned by the train, and by clustering the dictionary so that most atoms are not scored.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_01-dictionary-matching_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/01-dictionary-matching`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dictionary matching</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="You map T1 from a two-block MP2RAGE acquisition in two ways, by interpolating along the signal curve and by dictionary matching, and sweep the number of points on the curve to see which method depends on it.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_02-lookup-table_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/02-lookup-table`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MP2RAGE lookup table</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="You map T2 from a multi-echo spin echo by nonlinear least squares and compare the result with dictionary matching on the same slice.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_03-nonlinear-least-squares_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">T2 mapping by nonlinear least squares</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="You map T1, T2 and proton density in a brain slice with PERK, vary the size of the regression, and read the uncertainty it reports against the Cramér-Rao bound.">

.. only:: html

  .. image:: /generated/autoexamples/02-parameter-inference/images/thumb/sphx_glr_04-perk_thumb.png
    :alt:

  :doc:`/generated/autoexamples/02-parameter-inference/04-perk`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PERK: kernel ridge regression</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /generated/autoexamples/02-parameter-inference/01-dictionary-matching
   /generated/autoexamples/02-parameter-inference/02-lookup-table
   /generated/autoexamples/02-parameter-inference/03-nonlinear-least-squares
   /generated/autoexamples/02-parameter-inference/04-perk

