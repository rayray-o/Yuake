export type OneEuroOptions = {
  minCutoff: number;
  beta: number;
  dCutoff: number;
};

function smoothingFactor(
  cutoff: number,
  deltaTime: number
): number {
  if (
    !Number.isFinite(deltaTime) ||
    deltaTime <= 0
  ) {
    return 1;
  }

  const tau =
    1 /
    (2 *
      Math.PI *
      cutoff);

  return 1 /
    (
      1 +
      tau /
        deltaTime
    );
}

class LowPassFilter {
  private initialized =
    false;

  private value = 0;

  filter(
    input: number,
    alpha: number
  ): number {
    if (!this.initialized) {
      this.value = input;
      this.initialized = true;
      return input;
    }

    this.value =
      alpha * input +
      (1 - alpha) *
        this.value;

    return this.value;
  }

  reset(value = 0) {
    this.initialized = false;
    this.value = value;
  }
}

export class OneEuroFilter {
  private readonly options: OneEuroOptions;

  private readonly valueFilter =
    new LowPassFilter();

  private readonly derivativeFilter =
    new LowPassFilter();

  private previousRaw =
    0;

  private previousTime =
    0;

  private initialized =
    false;

  constructor(
    options: OneEuroOptions
  ) {
    this.options = options;
  }

  filter(
    value: number,
    timestampSeconds: number
  ): number {
    if (!this.initialized) {
      this.initialized = true;
      this.previousRaw = value;
      this.previousTime =
        timestampSeconds;

      return this.valueFilter.filter(
        value,
        1
      );
    }

    const deltaTime =
      Math.max(
        timestampSeconds -
          this.previousTime,
        0.000001
      );

    const rawDerivative =
      (
        value -
        this.previousRaw
      ) /
      deltaTime;

    const derivativeAlpha =
      smoothingFactor(
        this.options.dCutoff,
        deltaTime
      );

    const derivative =
      this.derivativeFilter.filter(
        rawDerivative,
        derivativeAlpha
      );

    const cutoff =
      this.options.minCutoff +
      this.options.beta *
        Math.abs(
          derivative
        );

    const alpha =
      smoothingFactor(
        cutoff,
        deltaTime
      );

    const result =
      this.valueFilter.filter(
        value,
        alpha
      );

    this.previousRaw =
      value;

    this.previousTime =
      timestampSeconds;

    return result;
  }

  reset() {
    this.initialized =
      false;

    this.previousRaw =
      0;

    this.previousTime =
      0;

    this.valueFilter.reset();
    this.derivativeFilter.reset();
  }
}

export class OneEuroPointFilter {
  readonly x: OneEuroFilter;

  readonly y: OneEuroFilter;

  constructor(
    options: OneEuroOptions
  ) {
    this.x =
      new OneEuroFilter(
        options
      );

    this.y =
      new OneEuroFilter(
        options
      );
  }

  filter(
    point: {
      x: number;
      y: number;
    },
    timestampSeconds: number
  ) {
    return {
      x: this.x.filter(
        point.x,
        timestampSeconds
      ),
      y: this.y.filter(
        point.y,
        timestampSeconds
      )
    };
  }

  reset() {
    this.x.reset();
    this.y.reset();
  }
        }
